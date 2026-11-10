"""TrustPay AI backend API.

Run: uvicorn app.main:create_app --factory --port 8000

Flow:
  1. POST /api/analyze or /api/analyze/url        -> agents produce a risk assessment
  2. POST /api/assessments/{id}/decision          -> buyer approves/rejects (gate enforced here)
  3. POST /api/assessments/{id}/orders            -> PayPal order created, returns approve_url
  4. Buyer approves on PayPal -> GET /api/paypal/return captures and verifies the payment
  5. POST /api/paypal/webhook keeps status in sync (verified with PayPal)
"""
from __future__ import annotations

import json
import logging
import secrets
import time
from collections import defaultdict, deque
from typing import Optional
from urllib.parse import urlencode

import httpx
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from pydantic import BaseModel, Field, ValidationError
from starlette.middleware.base import BaseHTTPMiddleware

from .sponsors import sponsor_router
from .agents import assess
from .config import PAYPAL_CURRENCIES, Settings, format_amount
from .db import Store
from .ingest import IngestError, listing_from_url
from .models import CheckoutOptions, Assessment, Decision, Listing, RiskLevel, UrlAnalyzeRequest
from .paypal import PayPalClient, PayPalError, localized_approval_url

log = logging.getLogger("trustpay")

CORS_METHODS = ("GET", "POST", "OPTIONS")
CORS_HEADERS = ("Authorization", "Content-Type", "X-API-Key")


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Baseline browser hardening headers for API and optional static UI."""

    def __init__(self, app, *, is_production: bool):
        super().__init__(app)
        self.is_production = is_production

    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Permissions-Policy", "camera=(), microphone=(), geolocation=()")
        if self.is_production:
            response.headers.setdefault("Strict-Transport-Security", "max-age=31536000; includeSubDomains")
        return response

OVERRIDABLE = {
    "title", "description", "category", "price", "currency", "return_policy", "shipping_days",
    "seller.name", "seller.account_age_days", "seller.rating", "seller.review_count",
    "seller.location", "seller.verified", "seller.paypal_email",
}


OPEN_ORDER_STATUSES = {"CREATED", "PAYER_ACTION_REQUIRED", "APPROVED"}
FINAL_STATUSES = {"COMPLETED", "PENDING", "REFUNDED", "REVERSED"}


class RateLimiter:
    """Sliding one-minute window per client IP. In-memory, so per instance."""

    def __init__(self, per_minute: int):
        self.per_minute = per_minute
        self.hits: dict[str, deque] = defaultdict(deque)

    def check(self, key: str) -> None:
        if self.per_minute <= 0:
            return
        now = time.monotonic()
        q = self.hits[key]
        while q and now - q[0] > 60:
            q.popleft()
        if len(q) >= self.per_minute:
            raise HTTPException(429, "Too many analysis requests; try again in a minute", headers={"Retry-After": "60"})
        q.append(now)


class RefundRequest(BaseModel):
    reason: str = Field(..., min_length=3, max_length=255)


def create_app(settings: Optional[Settings] = None, paypal_transport: Optional[httpx.AsyncBaseTransport] = None) -> FastAPI:
    logging.basicConfig(level=logging.INFO)
    settings = settings or Settings.from_env()
    settings.validate()  # no demo mode: refuse to start without PayPal + LLM credentials

    store = Store(settings.db_path)
    paypal = PayPalClient(settings, transport=paypal_transport)

    docs = "/docs" if settings.docs_enabled else None
    app = FastAPI(
        title="TrustPay AI",
        version="2.0.0",
        description="Pre-payment risk review with a human approval gate and PayPal checkout.",
        docs_url=docs,
        redoc_url="/redoc" if settings.docs_enabled else None,
        openapi_url="/openapi.json" if settings.docs_enabled else None,
    )
    app.state.settings, app.state.store, app.state.paypal = settings, store, paypal
    app.add_middleware(SecurityHeadersMiddleware, is_production=settings.is_production)
    if settings.allowed_hosts:
        app.add_middleware(TrustedHostMiddleware, allowed_hosts=list(settings.allowed_hosts))
    if settings.allowed_origins:
        # Explicit methods/headers only — avoid wildcard CORS in browser clients.
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(settings.allowed_origins),
            allow_methods=list(CORS_METHODS),
            allow_headers=list(CORS_HEADERS),
            allow_credentials=False,
            max_age=600,
        )

    limiter = RateLimiter(settings.rate_limit_per_min)

    def api_key_ok(provided: Optional[str]) -> bool:
        expected = settings.api_key
        if not expected:
            return True
        if not provided:
            return False
        # compare_digest requires equal-length strings; pad via secrets only when lengths match.
        return secrets.compare_digest(provided, expected) if len(provided) == len(expected) else False

    async def require_api_key(x_api_key: Optional[str] = Header(None)) -> None:
        if settings.api_key and not api_key_ok(x_api_key):
            raise HTTPException(401, "Invalid or missing X-API-Key")

    async def rate_limit(request: Request) -> None:
        # Each analysis makes several LLM calls, so these routes are throttled per client.
        # Only trust X-Forwarded-For when the edge proxy is known (TRUST_PROXY=true).
        if settings.trust_proxy:
            forwarded = request.headers.get("x-forwarded-for", "")
            client_key = forwarded.split(",")[0].strip() or (request.client.host if request.client else "unknown")
        else:
            client_key = request.client.host if request.client else "unknown"
        limiter.check(client_key)

    def finish(a: Optional[Assessment], status: str, **extra) -> object:
        """Where the buyer's browser lands after PayPal: the frontend if configured, else JSON."""
        body = {"assessment_id": a.id if a else None, "status": status, **extra}
        if settings.frontend_url:
            query = urlencode({k: v for k, v in body.items() if v is not None and not isinstance(v, (list, dict))})
            return RedirectResponse(f"{settings.frontend_url}/?{query}", status_code=303)
        return body

    @app.exception_handler(PayPalError)
    async def paypal_error_handler(_: Request, exc: PayPalError):
        content = {"detail": "PayPal request failed", "paypal_status": exc.status, "debug_id": exc.debug_id}
        if not settings.is_production:
            content["detail"] = str(exc)
            content["paypal_details"] = exc.details
        return JSONResponse(status_code=502, content=content)

    def get_or_404(assessment_id: str) -> Assessment:
        a = store.get(assessment_id)
        if not a:
            raise HTTPException(404, "Assessment not found")
        return a

    async def run_assessment(listing: Listing, source: str, missing: list[str]) -> Assessment:
        a = await assess(listing, settings)
        a.source, a.missing_fields = source, missing
        store.save(a)
        store.log(a.id, "assessed", {"score": a.overall_score, "level": a.risk_level.value, "ai_status": a.ai_status, "source": source, "url": listing.url})
        return a

    async def capture_and_verify(a: Assessment, order_id: str) -> dict:
        result = await paypal.capture_order(order_id)
        expected = {"value": format_amount(a.listing.price, a.listing.currency), "currency_code": a.listing.currency}
        got = result.get("amount") or {}
        mismatches = []
        if got.get("value") != expected["value"] or got.get("currency_code") != expected["currency_code"]:
            mismatches.append(f"amount {got} != expected {expected}")
        if result.get("custom_id") and not str(result["custom_id"]).startswith(f"trustpay:{a.id}:"):
            mismatches.append("custom_id does not match assessment")
        a.payment_status = result.get("capture_status") or result.get("order_status")
        a.capture_id = result.get("capture_id")
        if mismatches:
            a.payment_status = "VERIFICATION_FAILED"
        store.save(a)
        store.log(a.id, "paypal_captured", {**result, "verification_errors": mismatches})
        return {**result, "assessment_id": a.id, "verified": not mismatches, "verification_errors": mismatches}

    # ---------------------------------------------------------------- health
    @app.get("/api/health")
    async def health(deep: bool = False, x_api_key: Optional[str] = Header(None)):
        body = {
            "status": "ok",
            "paypal_env": settings.paypal_env,
            "llm": f"{settings.llm_provider}:{settings.llm_model}",
            "webhooks": bool(settings.paypal_webhook_id),
            "env": settings.trustpay_env,
        }
        if deep:
            # Deep checks probe PayPal with server credentials — require the access key when configured.
            if settings.api_key and not api_key_ok(x_api_key):
                raise HTTPException(401, "Invalid or missing X-API-Key")
            try:
                await paypal.verify_credentials()
                body["paypal_auth"] = "ok"
            except PayPalError as exc:
                body["paypal_auth"] = "failed"
                body["status"] = "degraded"
                if not settings.is_production:
                    body["paypal_auth_detail"] = str(exc)
        return body

    @app.get("/api/config")
    def public_config():
        """Public values a frontend needs to load the PayPal JS SDK. The client id is not a secret."""
        return {
            "paypal_client_id": settings.paypal_client_id,
            "paypal_env": settings.paypal_env,
            "currency": settings.currency,
            "api_key_required": bool(settings.api_key),
        }

    # --------------------------------------------------------------- analyze
    @app.post("/api/analyze", response_model=Assessment, dependencies=[Depends(require_api_key), Depends(rate_limit)])
    async def analyze(listing: Listing):
        missing = [f for f, v in (("market_price", listing.market_price), ("seller.account_age_days", listing.seller.account_age_days)) if v is None]
        return await run_assessment(listing, "manual", missing)

    @app.post("/api/analyze/url", response_model=Assessment, dependencies=[Depends(require_api_key), Depends(rate_limit)])
    async def analyze_url(req: UrlAnalyzeRequest):
        try:
            listing, missing = await listing_from_url(req.url, settings)
        except IngestError as exc:
            raise HTTPException(422, str(exc)) from exc
        data = listing.model_dump()
        for key, value in req.overrides.items():
            if key not in OVERRIDABLE:
                raise HTTPException(422, f"Field '{key}' can't be overridden")
            target = data["seller"] if key.startswith("seller.") else data
            target[key.removeprefix("seller.")] = value
            if key in missing:
                missing.remove(key)
        if req.market_price:
            data["market_price"] = req.market_price
            missing = [m for m in missing if m != "market_price"]
        data["seller_messages"] = req.seller_messages
        try:
            listing = Listing.model_validate(data)
        except ValidationError as exc:
            raise HTTPException(422, exc.errors()) from exc
        return await run_assessment(listing, "url", missing)

    # ----------------------------------------------------------- assessments
    @app.get("/api/assessments", response_model=list[Assessment], dependencies=[Depends(require_api_key)])
    def list_assessments(limit: int = 20):
        return store.recent(max(1, min(limit, 100)))

    @app.get("/api/assessments/{assessment_id}", response_model=Assessment, dependencies=[Depends(require_api_key)])
    def get_assessment(assessment_id: str):
        return get_or_404(assessment_id)

    @app.get("/api/assessments/{assessment_id}/audit", dependencies=[Depends(require_api_key)])
    def get_audit(assessment_id: str):
        get_or_404(assessment_id)
        return store.audit(assessment_id)

    @app.post("/api/assessments/{assessment_id}/decision", response_model=Assessment, dependencies=[Depends(require_api_key)])
    def decide(assessment_id: str, body: Decision):
        a = get_or_404(assessment_id)
        if a.paypal_order_id:
            raise HTTPException(409, "A PayPal order already exists for this assessment; decision is locked")
        if body.approve:
            if a.requires_override and not body.acknowledge_risk:
                raise HTTPException(409, f"{a.risk_level.value} risk: set acknowledge_risk=true to approve")
            if a.risk_level == RiskLevel.CRITICAL and len((body.note or "").strip()) < 10:
                raise HTTPException(409, "CRITICAL risk: a note of 10+ characters explaining why you're proceeding is required")
        a.decision = "approved" if body.approve else "rejected"
        store.save(a)
        store.log(a.id, f"user_{a.decision}", {"acknowledged": body.acknowledge_risk, "note": body.note})
        return a

    # --------------------------------------------------------------- payment
    @app.post("/api/assessments/{assessment_id}/orders", dependencies=[Depends(require_api_key)])
    async def create_order(assessment_id: str, body: Optional[CheckoutOptions] = None):
        a = get_or_404(assessment_id)
        if a.decision != "approved":
            raise HTTPException(403, "Payment blocked: the buyer has not approved this assessment")
        if a.payment_status in FINAL_STATUSES:
            raise HTTPException(409, f"Payment already {a.payment_status.lower()}")
        if a.listing.currency.upper() not in PAYPAL_CURRENCIES:
            raise HTTPException(422, f"PayPal doesn't support {a.listing.currency}; re-run the analysis with a supported currency")
        requested_locale = (body.locale if body else None) or settings.paypal_locale
        if a.paypal_order_id and a.payment_status in OPEN_ORDER_STATUSES:
            if body and body.locale and a.checkout_locale != requested_locale:
                raise HTTPException(409, "This order already exists with a different or unknown checkout language. Start a fresh assessment to choose another language. No duplicate payment was created.")
            # Buyer came back before paying: hand out the same order instead of creating a second one.
            return {"assessment_id": a.id, "order_id": a.paypal_order_id, "status": a.payment_status,
                    "approve_url": localized_approval_url(a.paypal_approve_url, a.checkout_locale or requested_locale), "reused": True, "locale": a.checkout_locale}
        a.order_attempts += 1
        payee = a.listing.seller.paypal_email
        order = await paypal.create_order(
            amount=a.listing.price,
            currency=a.listing.currency.upper(),
            description=a.listing.title,
            assessment_id=a.id,
            risk_level=a.risk_level.value,
            payee_email=payee,
            attempt=a.order_attempts,
            locale=(body.locale if body else None) or settings.paypal_locale,
        )
        a.checkout_locale = (body.locale if body else None) or settings.paypal_locale
        a.paypal_order_id, a.paypal_approve_url, a.payment_status = order["id"], order["approve_url"], order["status"]
        store.save(a)
        store.log(a.id, "paypal_order_created", {**order, "attempt": a.order_attempts, "payee": payee or "app account"})
        return {"assessment_id": a.id, "order_id": order["id"], "status": order["status"], "approve_url": order["approve_url"],
                "reused": False, "payee": payee, "locale": a.checkout_locale}

    @app.post("/api/assessments/{assessment_id}/capture", dependencies=[Depends(require_api_key)])
    async def capture(assessment_id: str):
        a = get_or_404(assessment_id)
        if a.decision != "approved" or not a.paypal_order_id:
            raise HTTPException(403, "No approved PayPal order for this assessment")
        if a.payment_status == "COMPLETED":
            raise HTTPException(409, "Already captured")
        return await capture_and_verify(a, a.paypal_order_id)

    @app.get("/api/paypal/return")
    async def paypal_return(token: str):
        """PayPal redirects the buyer here after approval; `token` is the order id."""
        a = store.get_by_order(token)
        if not a or a.decision != "approved":
            raise HTTPException(404, "Unknown or unapproved order")
        if a.payment_status == "COMPLETED":
            return finish(a, "COMPLETED", already_captured=True)
        result = await capture_and_verify(a, token)
        return finish(a, a.payment_status or "UNKNOWN", order_id=token, verified=result["verified"],
                      capture_id=result.get("capture_id"), capture_status=result.get("capture_status"),
                      verification_errors=result["verification_errors"])

    @app.get("/api/paypal/cancel")
    def paypal_cancel(token: str):
        a = store.get_by_order(token)
        if not a:
            raise HTTPException(404, "Unknown order")
        store.log(a.id, "buyer_cancelled_at_paypal", {"order_id": token})
        return finish(a, "CANCELLED_BY_BUYER", order_id=token)

    @app.post("/api/assessments/{assessment_id}/refund", dependencies=[Depends(require_api_key)])
    async def refund(assessment_id: str, body: RefundRequest):
        a = get_or_404(assessment_id)
        if not a.capture_id or a.payment_status != "COMPLETED":
            raise HTTPException(409, "Nothing to refund: no completed capture")
        result = await paypal.refund_capture(a.capture_id, body.reason)
        if result.get("status") == "COMPLETED":
            a.payment_status = "REFUNDED"
        store.save(a)
        store.log(a.id, "paypal_refund", {**result, "reason": body.reason})
        return result

    # --------------------------------------------------------------- webhook
    @app.post("/api/paypal/webhook")
    async def webhook(request: Request):
        raw = (await request.body()).decode("utf-8", errors="replace")
        try:
            event = json.loads(raw)
        except ValueError:
            raise HTTPException(400, "Body is not JSON")
        if not isinstance(event, dict) or not await paypal.verify_webhook(dict(request.headers), raw):
            raise HTTPException(400, "Webhook signature verification failed")
        event_id, event_type = event.get("id", ""), event.get("event_type", "")
        if not store.mark_event(event_id, event_type):
            return {"status": "duplicate"}

        resource = event.get("resource") or {}
        a = find_assessment(resource)
        if not a:
            return {"status": "ignored", "reason": "not a TrustPay order"}
        store.log(a.id, "paypal_webhook", {"event_id": event_id, "type": event_type})

        # Buyer approved on PayPal but never came back through /api/paypal/return (closed the tab):
        # capture server-side so an approved payment isn't left hanging.
        if event_type == "CHECKOUT.ORDER.APPROVED":
            if a.decision == "approved" and resource.get("id") == a.paypal_order_id and a.payment_status not in FINAL_STATUSES | {"VERIFICATION_FAILED"}:
                try:
                    result = await capture_and_verify(a, a.paypal_order_id)
                except PayPalError as exc:
                    store.log(a.id, "webhook_capture_failed", {"error": str(exc), "debug_id": exc.debug_id})
                    return {"status": "capture_failed"}
                return {"status": "captured", "verified": result["verified"]}
            return {"status": "processed"}

        status_map = {
            "PAYMENT.CAPTURE.COMPLETED": "COMPLETED",
            "PAYMENT.CAPTURE.PENDING": "PENDING",
            "PAYMENT.CAPTURE.DENIED": "DENIED",
            "PAYMENT.CAPTURE.REFUNDED": "REFUNDED",
            "PAYMENT.CAPTURE.REVERSED": "REVERSED",
        }
        new_status = status_map.get(event_type)
        # Webhooks can arrive out of order: a status may only move forward (PENDING -> COMPLETED/DENIED ->
        # REFUNDED/REVERSED). A failed amount verification can only be followed by a refund or reversal.
        rank = {"PENDING": 1, "COMPLETED": 2, "DENIED": 2, "REFUNDED": 3, "REVERSED": 3, "VERIFICATION_FAILED": 2.5}
        if new_status and rank[new_status] >= rank.get(a.payment_status or "", 0):
            a.payment_status = new_status
            if event_type in ("PAYMENT.CAPTURE.COMPLETED", "PAYMENT.CAPTURE.PENDING", "PAYMENT.CAPTURE.DENIED") and resource.get("id"):
                a.capture_id = resource["id"]
            store.save(a)
        return {"status": "processed"}

    def find_assessment(resource: dict) -> Optional[Assessment]:
        custom = resource.get("custom_id") or ((resource.get("purchase_units") or [{}])[0].get("custom_id")) or ""
        if custom.startswith("trustpay:"):
            return store.get(custom.split(":")[1])
        # Refund resources may not carry custom_id; their "up" link points at the capture.
        for link in resource.get("links") or []:
            href = link.get("href", "")
            if link.get("rel") == "up" and "/captures/" in href:
                return store.get_by_capture(href.rstrip("/").split("/")[-1])
        return None

    app.include_router(sponsor_router(settings, store, require_api_key, rate_limit))
    from pathlib import Path
    from fastapi.staticfiles import StaticFiles
    frontend = Path(__file__).resolve().parents[1] / "frontend_dist"
    if frontend.is_dir():
        app.mount("/", StaticFiles(directory=frontend, html=True), name="frontend")
    return app
