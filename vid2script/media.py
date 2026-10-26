import asyncio
import base64
import json
import math
import re
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import srt
from PIL import Image
from pydantic import BaseModel, Field

from .config import Settings


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

    @property
    def interval(self) -> float:
        return 1.0 if self.duration <= 60 else 2.0


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
    stream = next(s for s in streams if s["codec_type"] == "video" and not s.get("disposition", {}).get("attached_pic"))
    duration = float(data["format"]["duration"])
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("视频时长无效")
    subtitle = next((s["index"] for s in streams if s["codec_type"] == "subtitle"
                     and s["codec_name"] in {"subrip", "ass", "ssa", "mov_text", "webvtt", "text"}), None)
    return Video(path, duration, stream["width"], stream["height"],
                 any(s["codec_type"] == "audio" for s in streams), subtitle)


def timestamps(video: Video, start: float, end: float) -> list[float]:
    first = math.ceil((start - 1e-6) / video.interval)
    stop = math.ceil((min(end, video.duration) - 1e-6) / video.interval)
    return [round(i * video.interval, 3) for i in range(max(0, first), stop)]


class Media:
    def __init__(self, video: Video, root: Path, settings: Settings):
        self.video, self.root, self.settings = video, root, settings
        self.frames = root / "frames"
        self.frames.mkdir()

    async def images(self, start: float, end: float, crop: list[float] | None = None) -> tuple[list[dict], list[str]]:
        times = timestamps(self.video, start, end)
        if not times:
            if not 0 <= start < end <= self.video.duration + 0.001:
                raise ValueError("请求范围无效")
            times = [math.floor(start / self.video.interval) * self.video.interval]
        if len(times) > self.settings.max_images:
            raise ValueError("补充观察范围超过单次图片上限，请缩小范围")
        for path in self.frames.glob("*.jpg"):
            path.unlink()
        # fps 的起点始终位于原视频统一采样时间点，切分片段不会增加采样密度。
        args = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-ss", str(times[0]),
                "-i", str(self.video.path), "-map", "0:v:0", "-an", "-sn",
                "-vf", f"fps=1/{self.video.interval}:round=near", "-frames:v", str(len(times)),
                "-q:v", "2", "-y", str(self.frames / "%04d.jpg")]
        await run(*args)
        paths = sorted(self.frames.glob("*.jpg"))
        if len(paths) != len(times):
            raise RuntimeError(f"抽帧数量异常：预期 {len(times)}，实际 {len(paths)}")
        content, ids = [], []
        for timestamp, path in zip(times, paths, strict=True):
            frame_id = f"f{round(timestamp * 1000):010d}"
            ids.append(frame_id)
            data = compress(path, self.settings, crop)
            content.extend([
                {"type": "text", "text": f"frame_id={frame_id}; timestamp={timestamp:.3f}s"},
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(data).decode()}},
            ])
        return content, ids

    async def audio(self, start: float, end: float) -> Path:
        target = self.root / "audio.wav"
        await run("ffmpeg", "-hide_banner", "-loglevel", "error", "-ss", str(start),
                  "-i", str(self.video.path), "-t", str(end - start), "-vn", "-ac", "1", "-ar", "16000",
                  "-c:a", "pcm_s16le", "-y", str(target))
        return target

    async def silence_cut(self, target: float) -> float:
        start = max(0, target - 5)
        end = min(self.video.duration, target + 5)
        log = await run("ffmpeg", "-hide_banner", "-loglevel", "info", "-ss", str(start),
                        "-i", str(self.video.path), "-t", str(end - start), "-vn",
                        "-af", "silencedetect=noise=-35dB:d=0.35", "-f", "null", "-")
        ranges = re.findall(r"silence_end: ([\d.]+) \| silence_duration: ([\d.]+)", log)
        cuts = [start + float(stop) - float(length) / 2 for stop, length in ranges]
        return min(cuts, key=lambda value: abs(value - target)) if cuts else target

    async def subtitles(self, external: Path | None) -> list[Cue] | None:
        if external is None and self.video.subtitle_index is None:
            return None
        if external:
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


def compress(path: Path, settings: Settings, crop: list[float] | None = None) -> bytes:
    with Image.open(path) as original:
        picture = original.convert("RGB")
    if crop:
        x, y, width, height = crop
        if min(x, y) < 0 or min(width, height) <= 0 or x + width > 1 or y + height > 1:
            raise ValueError("裁剪范围必须位于图片内，使用 0 到 1 的相对坐标")
        picture = picture.crop((round(x * picture.width), round(y * picture.height),
                                round((x + width) * picture.width), round((y + height) * picture.height)))
    picture.thumbnail((settings.image_long_edge, settings.image_long_edge), Image.Resampling.LANCZOS)
    for quality in (80, 65, 50):
        output = BytesIO()
        picture.save(output, format="JPEG", quality=quality, optimize=True)
        if output.tell() <= settings.image_max_bytes:
            return output.getvalue()
    raise ValueError("图片压缩后仍超过单图字节预算，请调整 IMAGE_LONG_EDGE 或 IMAGE_MAX_BYTES")
