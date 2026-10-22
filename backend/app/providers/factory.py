"""Explicit model-provider selection. Provider failures never trigger a fallback."""
import os

from .base import ProviderConfigurationError

DEFAULT_MODELS = {"gemma": "gemma-4-31b-it", "demo": "deterministic-sequence"}


def provider_settings() -> dict[str, str | bool]:
    selected = os.getenv("MODEL_PROVIDER", "gemma").strip().lower()
    demo_mode = selected == "demo" or os.getenv("DEMO_MODE", "false").strip().lower() in {"1", "true", "yes", "on"}
    provider = "demo" if demo_mode else selected
    default_model = DEFAULT_MODELS.get(provider, DEFAULT_MODELS["gemma"])
    model = default_model if provider == "demo" else os.getenv("MODEL_NAME", default_model)
    if provider == "demo":
        api_status = "demo_active"
    elif provider == "gemma":
        api_status = "configured" if os.getenv("GEMMA_API_KEY") else "missing_credentials"
    else:
        api_status = "unsupported_provider"
    return {"model_provider": provider, "model_name": model, "demo_mode": demo_mode, "api_status": api_status}


async def provider_health() -> dict[str, str | bool]:
    """Return a verified provider status without exposing credentials."""
    settings = provider_settings()
    if settings["demo_mode"]:
        return {**settings, "provider_status": "demo_active"}
    if settings["model_provider"] != "gemma" or settings["api_status"] == "missing_credentials":
        return {**settings, "provider_status": "unavailable"}
    provider = None
    try:
        provider = create_provider("gemma")
        result = await provider.health_check()
        connected = result.get("status") == "available"
        status = "connected" if connected else "unavailable"
        return {**settings, "api_status": status, "provider_status": status}
    except Exception:
        return {**settings, "api_status": "unavailable", "provider_status": "unavailable"}
    finally:
        if provider is not None:
            await provider.aclose()


def create_provider(provider_name: str | None = None, scenario: str | None = None):
    name = (provider_name or os.getenv("MODEL_PROVIDER", "gemma")).strip().lower()
    if name == "gemma":
        from .gemma import GemmaProvider
        return GemmaProvider()
    if name == "demo":
        from .demo import DemoDecisionProvider
        return DemoDecisionProvider(scenario or "misfire")
    raise ProviderConfigurationError("Unsupported MODEL_PROVIDER. Choose gemma or demo")
