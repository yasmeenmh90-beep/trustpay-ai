"""End-to-end API flow against a fake PayPal API."""
import pytest

from app.agents import base, orchestrator
from app.config import ConfigError

from .conftest import make_settings
from .fixtures import CLEAN, MEDIUM, SCAM

WEBHOOK_HEADERS = {
    "PAYPAL-AUTH-ALGO": "SHA256withRSA", "PAYPAL-CERT-URL": "https://api.sandbox.paypal.com/cert",
    "PAYPAL-TRANSMISSION-ID": "t1", "PAYPAL-TRANSMISSION-SIG": "sig", "PAYPAL-TRANSMISSION-TIME": "2026-10-27T00:00:00Z",
}


def analyze(client, listing):
    r = client.post("/api/analyze", json=listing.model_dump())
    assert r.status_code == 200, r.text
    return r.json()


def approve_and_order(client, a, **decision):
    assert client.post(f"/api/assessments/{a['id']}/decision", json={"approve": True, **decision}).status_code == 200
    r = client.post(f"/api/assessments/{a['id']}/orders")
    assert r.status_code == 200, r.text
    return r.json()


@pytest.mark.parametrize("missing", ["paypal_client_id", "paypal_client_secret", "llm_api_key"])
def test_refuses_to_start_without_credentials(tmp_path, missing):
    with pytest.raises(ConfigError):
        make_settings(tmp_path, **{missing: ""}).validate()


def test_production_requires_api_key(tmp_path):
    with pytest.raises(ConfigError, match="TRUSTPAY_API_KEY"):
        make_settings(tmp_path, trustpay_env="production", api_key="").validate()


def test_production_requires_cors_when_frontend_set(tmp_path):
    with pytest.raises(ConfigError, match="ALLOWED_ORIGINS"):
        make_settings(
            tmp_path, trustpay_env="production", api_key="prod-key",
            public_base_url="https://api.example.com",
            frontend_url="https://app.example.com", allowed_origins=(),
        ).validate()


def test_production_same_origin_fly_skips_cors(tmp_path):
    make_settings(
        tmp_path, trustpay_env="production", api_key="prod-key",
        public_base_url="https://trustpay-ai.fly.dev",
        frontend_url="https://trustpay-ai.fly.dev", allowed_origins=(),
    ).validate()


def test_production_disables_openapi_docs(tmp_path, fake_paypal, llm_down):
    import httpx
    from fastapi.testclient import TestClient
    from app.main import create_app

    app = create_app(
        make_settings(tmp_path, trustpay_env="production", api_key="prod-key",
                      frontend_url="https://app.example.com",
                      allowed_origins=("https://app.example.com",)),
        paypal_transport=httpx.MockTransport(fake_paypal),
    )
    c = TestClient(app)
    assert c.get("/docs").status_code == 404
    assert c.get("/openapi.json").status_code == 404
    assert c.get("/api/health").headers.get("x-content-type-options") == "nosniff"
    assert c.get("/api/health?deep=true").status_code == 401
    assert c.get("/api/health?deep=true", headers={"X-API-Key": "prod-key"}).status_code == 200


def test_risk_tiers(client):
    assert analyze(client, CLEAN)["risk_level"] == "LOW"
    assert analyze(client, MEDIUM)["risk_level"] == "MEDIUM"
    scam = analyze(client, SCAM)
    assert scam["risk_level"] == "CRITICAL"
    signals = {f["signal"] for r in scam["reports"] for f in r["findings"]}
    assert {"Asks for unprotected payment", "Implausibly low price", "Attempt to manipulate AI review"} <= signals


def test_llm_outage_is_reported_not_hidden(client):
    a = analyze(client, SCAM)
    assert a["ai_status"] == "failed"
    assert "rules only" in a["summary"]
    assert a["risk_level"] == "CRITICAL"  # rules still catch it


def test_full_payment_flow_with_paypal_redirect(client, fake_paypal):
    a = analyze(client, CLEAN)
    order = approve_and_order(client, a)
    assert order["approve_url"].startswith("https://www.sandbox.paypal.com/checkoutnow")

    unit = fake_paypal.orders[order["order_id"]]
    assert unit["amount"] == {"currency_code": "USD", "value": "179.00"}
    assert unit["custom_id"] == f"trustpay:{a['id']}:LOW"
    sent = [r for r in fake_paypal.requests if r.url.path == "/v2/checkout/orders"][0]
    ctx = __import__("json").loads(sent.content)["payment_source"]["paypal"]["experience_context"]
    assert ctx["return_url"] == "https://trustpay.example/api/paypal/return"
    assert sent.headers["PayPal-Request-Id"] == f"order-{a['id']}-1"

    cap = client.get(f"/api/paypal/return?token={order['order_id']}&PayerID=P1").json()
    assert cap["verified"] and cap["capture_status"] == "COMPLETED"

    stored = client.get(f"/api/assessments/{a['id']}").json()
    assert stored["payment_status"] == "COMPLETED" and stored["capture_id"] == f"CAP-{order['order_id']}"
    events = [e["event"] for e in client.get(f"/api/assessments/{a['id']}/audit").json()]
    assert events == ["assessed", "user_approved", "paypal_order_created", "paypal_captured"]

    # idempotent return, no double order
    assert client.get(f"/api/paypal/return?token={order['order_id']}").json()["already_captured"]
    assert client.post(f"/api/assessments/{a['id']}/orders").status_code == 409


def test_amount_mismatch_fails_verification(client, fake_paypal):
    a = analyze(client, CLEAN)
    order = approve_and_order(client, a)
    fake_paypal.capture_value_override = "1.00"
    cap = client.post(f"/api/assessments/{a['id']}/capture").json()
    assert not cap["verified"]
    assert client.get(f"/api/assessments/{a['id']}").json()["payment_status"] == "VERIFICATION_FAILED"


def test_gate(client):
    a = analyze(client, CLEAN)
    assert client.post(f"/api/assessments/{a['id']}/orders").status_code == 403
    client.post(f"/api/assessments/{a['id']}/decision", json={"approve": False})
    assert client.post(f"/api/assessments/{a['id']}/orders").status_code == 403

    s = analyze(client, SCAM)
    url = f"/api/assessments/{s['id']}/decision"
    assert client.post(url, json={"approve": True}).status_code == 409
    assert client.post(url, json={"approve": True, "acknowledge_risk": True}).status_code == 409
    assert client.post(url, json={"approve": True, "acknowledge_risk": True, "note": "Known seller, met in person"}).status_code == 200


def test_decision_locked_after_order(client):
    a = analyze(client, CLEAN)
    approve_and_order(client, a)
    assert client.post(f"/api/assessments/{a['id']}/decision", json={"approve": False}).status_code == 409


def test_return_rejects_unknown_order(client):
    assert client.get("/api/paypal/return?token=NOPE").status_code == 404


def test_refund(client):
    a = analyze(client, CLEAN)
    assert client.post(f"/api/assessments/{a['id']}/refund", json={"reason": "not received"}).status_code == 409
    order = approve_and_order(client, a)
    client.get(f"/api/paypal/return?token={order['order_id']}")
    r = client.post(f"/api/assessments/{a['id']}/refund", json={"reason": "item not as described"})
    assert r.json()["status"] == "COMPLETED"
    assert client.get(f"/api/assessments/{a['id']}").json()["payment_status"] == "REFUNDED"


def test_webhook_verified_dedup_and_ordering(client, fake_paypal):
    a = analyze(client, CLEAN)
    order = approve_and_order(client, a)
    custom = f"trustpay:{a['id']}:LOW"
    completed = {"id": "WH-EVT-1", "event_type": "PAYMENT.CAPTURE.COMPLETED", "resource": {"id": "CAP9", "custom_id": custom}}
    assert client.post("/api/paypal/webhook", json=completed, headers=WEBHOOK_HEADERS).json()["status"] == "processed"
    assert client.post("/api/paypal/webhook", json=completed, headers=WEBHOOK_HEADERS).json()["status"] == "duplicate"

    late = {"id": "WH-EVT-2", "event_type": "CHECKOUT.ORDER.APPROVED", "resource": {"id": order["order_id"], "purchase_units": [{"custom_id": custom}]}}
    client.post("/api/paypal/webhook", json=late, headers=WEBHOOK_HEADERS)
    stored = client.get(f"/api/assessments/{a['id']}").json()
    assert stored["payment_status"] == "COMPLETED" and stored["capture_id"] == "CAP9"


def test_webhook_bad_signature(client, fake_paypal):
    fake_paypal.webhook_valid = False
    r = client.post("/api/paypal/webhook", json={"id": "x", "event_type": "PAYMENT.CAPTURE.COMPLETED"}, headers=WEBHOOK_HEADERS)
    assert r.status_code == 400
    assert client.post("/api/paypal/webhook", json={"id": "y"}).status_code == 400  # missing headers


def test_paypal_error_surfaces_debug_id(client, fake_paypal):
    a = analyze(client, CLEAN)
    approve_and_order(client, a)
    # capture an order the fake doesn't know -> PayPal 404 -> 502 with debug id
    fake_paypal.orders.clear()
    r = client.post(f"/api/assessments/{a['id']}/capture")
    assert r.status_code == 502
    assert r.json()["debug_id"] == "dbg-404" and r.json()["paypal_status"] == 404


def test_api_key(tmp_path, fake_paypal, llm_down):
    import httpx
    from fastapi.testclient import TestClient

    from app.main import create_app
    c = TestClient(create_app(make_settings(tmp_path, api_key="k"), paypal_transport=httpx.MockTransport(fake_paypal)))
    assert c.post("/api/analyze", json=CLEAN.model_dump()).status_code == 401
    assert c.post("/api/analyze", json=CLEAN.model_dump(), headers={"X-API-Key": "k"}).status_code == 200


def test_ai_ok_status(client, monkeypatch):
    async def ok(*_a, **_k):
        return {"additional_findings": [], "score_adjustment": 0, "summary": "AI summary", "recommendation": "AI reco"}
    monkeypatch.setattr(base, "complete_json", ok)
    monkeypatch.setattr(orchestrator, "complete_json", ok)
    a = analyze(client, CLEAN)
    assert a["ai_status"] == "ok" and a["summary"] == "AI summary"
