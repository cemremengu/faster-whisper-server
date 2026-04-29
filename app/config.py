from __future__ import annotations

from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

Device = Literal["auto", "cpu", "cuda"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="WHISPER_", env_file=None, extra="ignore"
    )

    model: str = "large-v3"
    device: Device = "auto"
    compute_type: str = "default"
    cpu_threads: int = 0
    num_workers: int = 1
    download_root: str | None = None

    def resolve_device(self) -> str:
        if self.device != "auto":
            return self.device
        try:
            import ctranslate2

            if ctranslate2.get_cuda_device_count() > 0:
                return "cuda"
        except Exception:
            pass
        return "cpu"


settings = Settings()
