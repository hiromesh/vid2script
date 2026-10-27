import os
from pathlib import Path
from typing import Literal

from dotenv import dotenv_values, set_key
from pydantic import BaseModel, ConfigDict, Field, SecretStr


class Settings(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    model: str
    base_url: str
    api_key: SecretStr
    thinking_mode: Literal["default", "enabled", "disabled"] = "default"
    compact_thinking_mode: Literal["default", "enabled", "disabled"] = "disabled"
    reasoning_effort: Literal["", "low", "high", "max"] = ""
    max_output_tokens: int = Field(default=8192, ge=2048)
    context_window_tokens: int = Field(default=131072, ge=16384)
    request_timeout_seconds: float = Field(default=240, gt=0)
    batch_seconds: float = Field(default=30, ge=2, le=60)
    max_request_bytes: int = Field(default=24 * 1024 * 1024, ge=1024 * 1024)
    video_long_edge: int = Field(default=640, ge=256, le=1280)
    video_fps: int = Field(default=3, ge=1, le=10)
    video_crf: int = Field(default=30, ge=18, le=40)
    video_max_bytes: int = Field(default=8 * 1024 * 1024, ge=100 * 1024)
    max_video_seconds: float = Field(default=30, ge=2, le=120)
    max_videos: int = Field(default=3, ge=1, le=8)
    max_turns_per_batch: int = Field(default=30, ge=4)
    asr_api_key: SecretStr = SecretStr("")
    asr_resource_id: str = "volc.bigasr.auc"
    asr_timeout_seconds: float = Field(default=900, gt=0)
    asr_poll_interval_seconds: float = Field(default=2, gt=0)
    asr_segment_seconds: float = Field(default=300, ge=10, le=600)
    oss_endpoint: str = ""
    oss_bucket: str = ""
    oss_access_key_id: SecretStr = SecretStr("")
    oss_access_key_secret: SecretStr = SecretStr("")

    @property
    def output_budget(self) -> int:
        return min(self.max_output_tokens, 8192, self.context_window_tokens // 4)

    @classmethod
    def load(cls, path: Path) -> "Settings":
        values = {**dotenv_values(path), **os.environ}
        names = {name: name.upper() for name in cls.model_fields}
        names.update({
            "asr_api_key": "VIDEO_ANALYSIS__ASR_API_KEY",
            "asr_resource_id": "VIDEO_ANALYSIS__ASR_RESOURCE_ID",
            "asr_timeout_seconds": "VIDEO_ANALYSIS__ASR_TIMEOUT_SECONDS",
            "oss_endpoint": "OSS__ENDPOINT", "oss_bucket": "OSS__BUCKET",
            "oss_access_key_id": "OSS__ACCESS_KEY_ID",
            "oss_access_key_secret": "OSS__ACCESS_KEY_SECRET",
        })
        return cls.model_validate({k: values[v] for k, v in names.items() if values.get(v)})

    def require_asr(self) -> None:
        required = ("asr_api_key", "oss_endpoint", "oss_bucket", "oss_access_key_id", "oss_access_key_secret")
        missing = []
        for key in required:
            value = getattr(self, key)
            if isinstance(value, SecretStr):
                value = value.get_secret_value()
            if not value:
                missing.append(key)
        if missing:
            raise ValueError(f"音频识别缺少配置：{', '.join(missing)}")


def import_services(source: Path, target: Path) -> list[str]:
    if not source.is_file():
        raise FileNotFoundError(source)
    incoming, existing = dotenv_values(source), dotenv_values(target)
    keys = [key for key in incoming if key.startswith(("OSS__", "ASR_", "VIDEO_ANALYSIS__ASR_"))]
    written = []
    for key in keys:
        if incoming[key] and not existing.get(key):
            set_key(target, key, incoming[key], quote_mode="always")
            written.append(key)
    if target.exists():
        target.chmod(0o600)
    return written
