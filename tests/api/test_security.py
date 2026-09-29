import hashlib
import hmac

import httpx
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from api.main import create_app
from channels.adapters.whatsapp.adapter import WhatsappAdapter
from channels.adapters.whatsapp.mapper import WhatsAppMessageMapper
from channels.config.channels import WhatsAppConfig
from channels.models.health import ChannelHealth
from channels.services.webhook import WebhookService

BODY = b'{"entry": []}'


def signed(body: bytes, secret: str) -> str:
    return "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()


def test_webhook_accepts_a_valid_signature():
    assert WebhookService(WhatsAppMessageMapper(), secret="s3cret").verify(BODY, signed(BODY, "s3cret"))


@pytest.mark.parametrize(
    "signature", [None, "", signed(BODY, "wrong"), signed(b"tampered", "s3cret"), "sha256=ünïcode"]
)
def test_webhook_rejects_bad_signatures(signature):
    assert not WebhookService(WhatsAppMessageMapper(), secret="s3cret").verify(BODY, signature)


def test_webhook_without_a_secret_rejects_everything():
    assert not WebhookService(WhatsAppMessageMapper(), secret=None).verify(BODY, signed(BODY, ""))


@pytest.mark.parametrize("secret", [None, ""])
def test_whatsapp_config_requires_a_webhook_secret(secret):
    fields = {"access_token": "t", "phone_number_id": "1"}
    if secret is not None:
        fields["webhook_secret"] = secret
    with pytest.raises(ValidationError):
        WhatsAppConfig(**fields)


@pytest.mark.parametrize(
    ("configured", "mode", "token", "ok"),
    [
        ("vt", "subscribe", "vt", True),
        ("vt", "subscribe", "nope", False),
        ("vt", "unsubscribe", "vt", False),
        ("vt", "subscribe", None, False),
        ("vt", "subscribe", "ünïcode", False),
        (None, "subscribe", "", False),
    ],
)
async def test_verify_challenge(configured, mode, token, ok):
    config = WhatsAppConfig(access_token="t", phone_number_id="1", webhook_secret="s", verify_token=configured)
    adapter = WhatsappAdapter(WebhookService(WhatsAppMessageMapper(), "s"), httpx.AsyncClient(), config, sender=None)
    try:
        assert adapter.verify_challenge(mode, token) is ok
    finally:
        await adapter.close()


class FakeChannel:
    async def health_check(self) -> ChannelHealth:
        return ChannelHealth(channel="whatsapp", ok=True, detail="+20 100")

    def verify_challenge(self, mode, token) -> bool:
        return token == "vt"


def client_with(api_key) -> TestClient:
    # No `with` block, so the lifespan (OCR, database, Graph API) never runs.
    app = create_app()
    app.state.whatsapp = FakeChannel()
    app.state.api_key = api_key
    return TestClient(app)


PROTECTED = ["/health", "/health/whatsapp"]


@pytest.mark.parametrize("path", PROTECTED)
def test_endpoints_accept_the_right_key(path):
    assert client_with("k3y").get(path, headers={"X-API-Key": "k3y"}).status_code == 200


@pytest.mark.parametrize("path", PROTECTED)
@pytest.mark.parametrize("headers", [{}, {"X-API-Key": ""}, {"X-API-Key": "wrong"}])
def test_endpoints_reject_missing_or_wrong_key(path, headers):
    assert client_with("k3y").get(path, headers=headers).status_code == 401


@pytest.mark.parametrize("path", PROTECTED)
def test_endpoints_are_locked_when_no_key_is_configured(path):
    assert client_with(None).get(path, headers={"X-API-Key": ""}).status_code == 401


def test_webhook_does_not_need_the_api_key():
    params = {"hub.mode": "subscribe", "hub.verify_token": "vt", "hub.challenge": "42"}
    response = client_with("k3y").get("/webhooks/whatsapp", params=params)
    assert response.status_code == 200
    assert response.text == "42"
