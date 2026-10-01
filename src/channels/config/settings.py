from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from channels.config.channels import ChannelsSettings as _ChannelsSettings


class _Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_nested_delimiter="__",
        extra="ignore",
    )

    project_id: str = "money-transaction"
    public_base_url: str = "http://localhost:8000"
    host: str = "0.0.0.0"
    port: int = 8000
    database_url: str = Field(default="sqlite+aiosqlite:///./data/smaia.db")
    database_migrate_on_startup: bool = True
    message_debounce_seconds: float = 5.0
    message_debounce_max_wait_seconds: float = 20.0
    generated_media_dir: Path = Path("data/generated_media")
    generated_media_url_path: str = "/media/generated"
    inbound_media_dir: Path = Path("data/inbound_media")
    inbound_media_url_path: str = "/media/inbound"
    inbound_media_max_bytes: int = 20 * 1024 * 1024
    ocr_enabled: bool = True
    # OCR worker processes = receipts read in parallel. Each one holds its own copy of the models.
    ocr_workers: int = Field(default=2, ge=1)
    # paddle | tesseract (Tesseract is CPU only)
    ocr_engine: Literal["paddle", "tesseract"] = "paddle"
    # auto = the GPU if there is one, else the CPU; or force cpu / gpu:0
    ocr_device: str = "auto"
    # Sent as the X-API-Key header on every endpoint except the webhook; unset = all locked.
    api_key: Optional[SecretStr] = None
    channels: _ChannelsSettings = Field(default_factory=_ChannelsSettings)


@lru_cache
def Settings() -> _Settings:
    return _Settings()


__all__ = ["Settings"]