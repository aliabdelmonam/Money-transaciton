from typing import Optional

from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict

from channels.models.exceptions import ChannelError


class TelegramConfig(BaseModel):
    bot_token: str
    api_base: str = "https://api.telegram.org"
    webhook_secret: Optional[str] = None


class WhatsAppConfig(BaseModel):
    access_token: str
    phone_number_id: str
    api_base: str = "https://graph.facebook.com"
    api_version: str = "v25.0"
    webhook_secret: Optional[str] = None
    verify_token: Optional[str] = None



class ChannelsSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="CHANNELS_",
        env_nested_delimiter="__",
        extra="ignore",
    )

    telegram: Optional[TelegramConfig] = None
    whatsapp: Optional[WhatsAppConfig] = None

    def configured_names(self) -> list[str]:
        return sorted(
            name
            for name in type(self).model_fields
            if getattr(self, name, None) is not None
        )

    def config_for(self, name: str) -> BaseModel:
        config = getattr(self, name, None)
        if config is None:
            raise ChannelError(f"Channel '{name}' is not configured")
        return config