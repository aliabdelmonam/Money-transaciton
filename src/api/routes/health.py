from fastapi import APIRouter, Request

from channels.models.health import ChannelHealth

router = APIRouter(prefix="/health", tags=["health"])


@router.get("")
async def health() -> dict:
    return {"status": "ok"}


@router.get("/whatsapp", response_model=ChannelHealth)
async def whatsapp_health(request: Request) -> ChannelHealth:
    """Check the access token and phone number id against the Graph API."""
    return await request.app.state.whatsapp.health_check()
