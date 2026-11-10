"""Runtime configuration from environment variables (.env supported).

There is no demo or simulated mode: PayPal credentials and an LLM key are
required, and the app refuses to start without them.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # python-dotenv is optional
    pass


# Currencies the PayPal REST APIs accept. HUF, JPY and TWD don't allow decimals.
PAYPAL_CURRENCIES = {
    "AUD", "BRL", "CAD", "CNY", "CZK", "DKK", "EUR", "HKD", "HUF", "ILS", "JPY", "MYR", "MXN", "TWD",
    "NZD", "NOK", "PHP", "PLN", "GBP", "SGD", "SEK", "CHF", "THB", "USD",
}
ZERO_DECIMAL_CURRENCIES = {"HUF", "JPY", "TWD"}


def format_amount(amount: float, currency: str) -> str:
    """Format an amount the way PayPal expects for this currency."""
    return f"{round(amount):d}" if currency.upper() in ZERO_DECIMAL_CURRENCIES else f"{amount:.2f}"


class ConfigError(RuntimeError):
    pass


@dataclass(frozen=True)
class Settings:
    paypal_client_id: str
    paypal_client_secret: str
    paypal_base_url: str = "https://api-m.sandbox.paypal.com"
    paypal_webhook_id: str = ""
    paypal_locale: str = "en-US"
    public_base_url: str = "http://localhost:8000"  # where PayPal redirects the buyer back to
    currency: str = "USD"

    llm_provider: str = "anthropic"  # anthropic | openai
    llm_api_key: str = ""
    llm_model: str = "claude-sonnet-5-5"
    openai_base_url: str = "https://api.openai.com/v1"
    llm_timeout_s: float = 30.0

    channel3_api_key: str = ""
    channel3_base_url: str = "https://api.trychannel3.com/v1"

    db_path: str = "trustpay.db"
    api_key: str = ""  # required when TRUSTPAY_ENV=production; require X-API-Key on /api routes
    allowed_origins: tuple[str, ...] = ()
    allowed_hosts: tuple[str, ...] = ()  # optional Host header allow-list (TrustedHostMiddleware)
    frontend_url: str = ""  # if set, PayPal return/cancel redirect the buyer here instead of returning JSON
    rate_limit_per_min: int = 20  # per client IP on /api/analyze*; 0 disables
    trustpay_env: str = "development"  # development | production
    trust_proxy: bool = False  # when True, use first X-Forwarded-For hop for rate limits
    enable_docs: bool | None = None  # None = auto (off in production); True/False overrides

    @classmethod
    def from_env(cls) -> "Settings":
        provider = os.getenv("LLM_PROVIDER", "anthropic").lower()
        key = os.getenv("ANTHROPIC_API_KEY" if provider == "anthropic" else "OPENAI_API_KEY", "")
        default_model = "claude-sonnet-5-5" if provider == "anthropic" else "gpt-4o-mini"
        docs_raw = os.getenv("TRUSTPAY_ENABLE_DOCS", "").strip().lower()
        enable_docs = None if docs_raw == "" else docs_raw in ("1", "true", "yes", "on")
        return cls(
            paypal_client_id=os.getenv("PAYPAL_CLIENT_ID", ""),
            paypal_client_secret=os.getenv("PAYPAL_CLIENT_SECRET", ""),
            paypal_base_url=os.getenv("PAYPAL_BASE_URL", "https://api-m.sandbox.paypal.com").rstrip("/"),
            paypal_locale=os.getenv("PAYPAL_LOCALE", "en-US"),
            paypal_webhook_id=os.getenv("PAYPAL_WEBHOOK_ID", ""),
            public_base_url=os.getenv("PUBLIC_BASE_URL", "http://localhost:8000").rstrip("/"),
            currency=os.getenv("CURRENCY", "USD"),
            llm_provider=provider,
            llm_api_key=key,
            llm_model=os.getenv("LLM_MODEL", default_model),
            openai_base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/"),
            llm_timeout_s=float(os.getenv("LLM_TIMEOUT_S", "30")),
            channel3_api_key=os.getenv("CHANNEL3_API_KEY", ""),
            channel3_base_url=os.getenv("CHANNEL3_BASE_URL", "https://api.trychannel3.com/v1").rstrip("/"),
            db_path=os.getenv("TRUSTPAY_DB", "trustpay.db"),
            api_key=os.getenv("TRUSTPAY_API_KEY", ""),
            allowed_origins=tuple(o.strip() for o in os.getenv("ALLOWED_ORIGINS", "").split(",") if o.strip()),
            allowed_hosts=tuple(h.strip() for h in os.getenv("ALLOWED_HOSTS", "").split(",") if h.strip()),
            frontend_url=os.getenv("FRONTEND_URL", "").rstrip("/"),
            rate_limit_per_min=int(os.getenv("RATE_LIMIT_PER_MIN", "20")),
            trustpay_env=os.getenv("TRUSTPAY_ENV", "development").strip().lower() or "development",
            trust_proxy=os.getenv("TRUST_PROXY", "").strip().lower() in ("1", "true", "yes", "on"),
            enable_docs=enable_docs,
        )

    @property
    def is_production(self) -> bool:
        return self.trustpay_env == "production"

    @property
    def docs_enabled(self) -> bool:
        if self.enable_docs is not None:
            return self.enable_docs
        return not self.is_production

    def validate(self) -> None:
        missing = []
        if self.trustpay_env not in ("development", "production"):
            raise ConfigError("TRUSTPAY_ENV must be 'development' or 'production'")
        if self.paypal_locale not in ("en-US", "ar-AE", "fr-FR"):
            raise ConfigError("PAYPAL_LOCALE must be en-US, ar-AE or fr-FR")
        if not self.paypal_client_id:
            missing.append("PAYPAL_CLIENT_ID")
        if not self.paypal_client_secret:
            missing.append("PAYPAL_CLIENT_SECRET")
        if self.currency.upper() not in PAYPAL_CURRENCIES:
            raise ConfigError(f"CURRENCY {self.currency} is not supported by PayPal")
        if self.llm_provider not in ("anthropic", "openai"):
            raise ConfigError("LLM_PROVIDER must be 'anthropic' or 'openai'")
        if not self.llm_api_key:
            missing.append("ANTHROPIC_API_KEY" if self.llm_provider == "anthropic" else "OPENAI_API_KEY")
        if self.is_production and not self.api_key:
            missing.append("TRUSTPAY_API_KEY")
        if missing:
            raise ConfigError("Missing required environment variables: " + ", ".join(missing))
        # Split UI (e.g. Vercel) needs CORS. Same-origin Docker/Fly (FRONTEND_URL == PUBLIC_BASE_URL) does not.
        if (
            self.is_production
            and self.frontend_url
            and not self.allowed_origins
            and self.frontend_url.rstrip("/") != self.public_base_url.rstrip("/")
        ):
            raise ConfigError(
                "Production with a separate FRONTEND_URL requires ALLOWED_ORIGINS "
                "(comma-separated browser origins for the Vercel/frontend host). "
                "Same-origin Fly/Docker can set FRONTEND_URL equal to PUBLIC_BASE_URL and leave ALLOWED_ORIGINS empty."
            )
        if self.is_production and "sandbox" not in self.paypal_base_url and not self.paypal_webhook_id:
            raise ConfigError("Live PayPal requires PAYPAL_WEBHOOK_ID so capture events stay in sync")

    @property
    def paypal_env(self) -> str:
        return "sandbox" if "sandbox" in self.paypal_base_url else "live"
