import asyncio
import base64
import json
import logging
import math
import re
from dataclasses import dataclass
from pathlib import Path

import srt
from pydantic import BaseModel, Field

from .config import Settings

log = logging.getLogger(__name__)


class Cue(BaseModel):
    id: str
    start: float = Field(ge=0)
    end: float = Field(ge=0)
    text: str
    speaker: str = ""
    source: str = "asr"


@dataclass(frozen=True)
class Video:
    path: Path
    duration: float
    width: int
    height: int
    audio: bool
    subtitle_index: int | None


@dataclass(frozen=True)
class Observation:
    """一段视频观察请求；extra_tokens 是该片段的保守 token 估算。"""

    content: list[dict]
    ids: list[str]
    extra_tokens: int = 0


async def run(*args: str) -> str:
    process = await asyncio.create_subprocess_exec(
        *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await process.communicate()
    except BaseException:
        if process.returncode is None:
            process.kill()
        await process.wait()
        raise
    if process.returncode:
        raise RuntimeError(f"{args[0]} 执行失败：{stderr.decode(errors='replace')[-1500:]}")
    return stdout.decode(errors="replace") or stderr.decode(errors="replace")


async def probe(path: Path) -> Video:
    if not path.is_file():
        raise FileNotFoundError(path)
    data = json.loads(await run("ffprobe", "-v", "error", "-show_format", "-show_streams", "-of", "json", str(path)))
    streams = data["streams"]
    stream = next((s for s in streams if s["codec_type"] == "video"
                   and not s.get("disposition", {}).get("attached_pic")), None)
    if stream is None:
        raise ValueError("视频没有可用的画面流")
    duration = float(data["format"]["duration"])
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("视频时长无效")
    subtitle = next((s["index"] for s in streams if s["codec_type"] == "subtitle"
                     and s["codec_name"] in {"subrip", "ass", "ssa", "mov_text", "webvtt", "text"}), None)
    return Video(path, duration, stream["width"], stream["height"],
                 any(s["codec_type"] == "audio" for s in streams), subtitle)


def estimate_video_tokens(seconds: float, width: int, height: int, fps: int) -> int:
    """按 Qwen 视频 token 的帧面积公式保守估算，并留出 15% 余量。"""
    return math.ceil(seconds * fps * width * height / 3136 * 1.15)


async def probe_size(path: Path) -> tuple[int, int]:
    data = json.loads(await run("ffprobe", "-v", "error", "-select_streams", "v:0",
                                "-show_entries", "stream=width,height", "-of", "json", str(path)))
    stream = data["streams"][0]
    return int(stream["width"]), int(stream["height"])


class Media:
    def __init__(self, video: Video, root: Path, settings: Settings):
        self.video, self.root, self.settings = video, root, settings
        self.clips = root / "clips"
        self.clips.mkdir(exist_ok=True)

    @property
    def max_span(self) -> float:
        return self.settings.max_video_seconds

    @property
    def min_span(self) -> float:
        return min(4.0, self.settings.max_video_seconds)

    @property
    def min_clip(self) -> float:
        # 服务端拒绝 1.5 秒片段；2 秒可用。
        return 2.0

    @property
    def review_span(self) -> float:
        return min(12.0, self.settings.max_video_seconds)

    def describe(self) -> dict:
        return {"clip_format": "mp4", "clip_long_edge": self.settings.video_long_edge,
                "clip_fps": self.settings.video_fps, "max_clip_seconds": self.settings.max_video_seconds,
                "clip_audio": "removed"}

    async def clip(self, start: float, end: float, crop: list[float] | None = None) -> Observation:
        start, end = max(0.0, start), min(self.video.duration, end)
        if end - start < self.min_clip:
            raise ValueError(f"视频观察范围至少需要 {self.min_clip:g} 秒")
        if end - start > self.settings.max_video_seconds + 0.001:
            raise ValueError(f"单个视频片段最长 {self.settings.max_video_seconds:g} 秒，请缩小观察范围")
        data, width, height = await self._encode(start, end, crop)
        clip_id = f"v{round(start * 1000):010d}-{round(end * 1000):010d}"
        url = "data:video/mp4;base64," + base64.b64encode(data).decode()
        content = [
            {"type": "text", "text": f"clip_id={clip_id}; range={start:.3f}-{end:.3f}s; audio=removed"},
            {"type": "video_url", "video_url": {"url": url}},
        ]
        tokens = estimate_video_tokens(end - start, width, height, self.settings.video_fps)
        log.info("视频片段 %s %.1f–%.1fs %sx%s %sKB → 约 %s tokens",
                 clip_id, start, end, width, height, round(len(data) / 1024), tokens)
        return Observation(content, [clip_id], tokens)

    async def _encode(self, start: float, end: float, crop: list[float] | None) -> tuple[bytes, int, int]:
        crop_filter = self._crop_filter(crop)
        attempts = []
        edge, fps, crf = self.settings.video_long_edge, self.settings.video_fps, self.settings.video_crf
        for _ in range(3):
            attempts.append((edge, fps, crf))
            edge, fps, crf = max(256, int(edge * 0.75) // 2 * 2), max(1, fps - 1), min(40, crf + 4)
        target = self.clips / "clip.mp4"
        for path in self.clips.glob("*.mp4"):
            path.unlink()
        for edge, fps, crf in attempts:
            filters = [crop_filter] if crop_filter else []
            filters.extend([f"scale='trunc(min({edge},iw)/2)*2':-2", f"fps={fps}", "setsar=1"])
            await run("ffmpeg", "-hide_banner", "-loglevel", "error", "-ss", str(start),
                      "-i", str(self.video.path), "-t", str(end - start), "-map", "0:v:0", "-an", "-sn",
                      "-vf", ",".join(filters), "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf),
                      "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-y", str(target))
            if target.stat().st_size <= self.settings.video_max_bytes:
                break
            target.unlink()
            log.info("视频片段超过字节预算，降低分辨率与帧率后重试")
        else:
            raise ValueError("视频片段压缩后仍超过单片段字节预算，请调整 VIDEO_LONG_EDGE 或 VIDEO_MAX_BYTES")
        width, height = await probe_size(target)
        return target.read_bytes(), width, height

    def _crop_filter(self, crop: list[float] | None) -> str:
        if not crop:
            return ""
        x, y, width, height = crop
        if min(x, y) < 0 or min(width, height) <= 0 or x + width > 1 or y + height > 1:
            raise ValueError("裁剪范围必须位于视频内，使用 0 到 1 的相对坐标")
        left = min(self.video.width - 2, int(x * self.video.width) // 2 * 2)
        top = min(self.video.height - 2, int(y * self.video.height) // 2 * 2)
        right = max(left + 2, min(self.video.width, int((x + width) * self.video.width)) // 2 * 2)
        bottom = max(top + 2, min(self.video.height, int((y + height) * self.video.height)) // 2 * 2)
        return f"crop={right - left}:{bottom - top}:{left}:{top}"

    async def audio(self, start: float, end: float) -> Path:
        target = self.root / "audio.wav"
        await run("ffmpeg", "-hide_banner", "-loglevel", "error", "-ss", str(start),
                  "-i", str(self.video.path), "-t", str(end - start), "-vn", "-ac", "1", "-ar", "16000",
                  "-c:a", "pcm_s16le", "-y", str(target))
        return target

    async def silence_cut(self, target: float) -> float:
        start = max(0, target - 5)
        end = min(self.video.duration, target + 5)
        log_text = await run("ffmpeg", "-hide_banner", "-loglevel", "info", "-ss", str(start),
                             "-i", str(self.video.path), "-t", str(end - start), "-vn",
                             "-af", "silencedetect=noise=-35dB:d=0.35", "-f", "null", "-")
        ranges = re.findall(r"silence_end: ([\d.]+) \| silence_duration: ([\d.]+)", log_text)
        cuts = [start + float(stop) - float(length) / 2 for stop, length in ranges]
        return min(cuts, key=lambda value: abs(value - target)) if cuts else target

    async def subtitles(self, external: Path | None) -> list[Cue] | None:
        if external is None and self.video.subtitle_index is None:
            return None
        if external is not None:
            source = external
            if not source.is_file():
                raise FileNotFoundError(source)
            text = await run("ffmpeg", "-v", "error", "-i", str(source), "-f", "srt", "-")
        else:
            text = await run("ffmpeg", "-v", "error", "-i", str(self.video.path),
                             "-map", f"0:{self.video.subtitle_index}", "-f", "srt", "-")
        cues = [Cue(id=f"sub_{i:06}", start=max(0, item.start.total_seconds()), end=item.end.total_seconds(),
                    text=item.content, source="subtitle") for i, item in enumerate(srt.parse(text))]
        if not cues:
            raise ValueError("所选字幕轨没有可读取的字幕")
        return cues
