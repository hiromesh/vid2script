import asyncio
import base64
import shutil
import subprocess
from pathlib import Path

import pytest

from vid2script.config import Settings
from vid2script.media import Media, probe


@pytest.fixture(scope="module")
def source(tmp_path_factory) -> Path:
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        pytest.skip("需要 ffmpeg 和 ffprobe")
    path = tmp_path_factory.mktemp("video") / "source.mp4"
    subprocess.run([
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i", "testsrc=size=320x240:rate=10:duration=3",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=3",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest",
        "-y", str(path),
    ], check=True, capture_output=True)
    return path


@pytest.fixture
def settings() -> Settings:
    return Settings(model="test", base_url="http://localhost/v1", api_key="test",
                    video_long_edge=320, video_fps=2, video_crf=30,
                    video_max_bytes=4 * 1024 * 1024, max_video_seconds=30)


@pytest.fixture
def video(source):
    return asyncio.run(probe(source))


def test_probe_reads_video_and_audio(video):
    assert video.duration == pytest.approx(3.0, abs=0.1)
    assert (video.width, video.height) == (320, 240)
    assert video.audio is True and video.subtitle_index is None


def test_clip_is_silent_base64_mp4(tmp_path: Path, video, settings):
    media = Media(video, tmp_path, settings)
    observation = asyncio.run(media.clip(0, 2))
    assert observation.ids == ["v0000000000-0000002000"]
    assert observation.content[0]["text"] == "clip_id=v0000000000-0000002000; range=0.000-2.000s; audio=removed"
    url = observation.content[1]["video_url"]["url"]
    assert url.startswith("data:video/mp4;base64,")
    data = base64.b64decode(url.split(",", 1)[1])
    assert b"ftyp" in data[:64]
    assert observation.extra_tokens > 0
    encoded = asyncio.run(probe(media.clips / "clip.mp4"))
    assert encoded.audio is False
    assert encoded.duration == pytest.approx(2.0, abs=0.2)
    assert encoded.width == 320


def test_clip_id_includes_end(tmp_path: Path, video, settings):
    media = Media(video, tmp_path, settings)
    observation = asyncio.run(media.clip(0.5, 2.5))
    assert observation.ids == ["v0000000500-0000002500"]


def test_clip_rejects_too_short_range(tmp_path: Path, video, settings):
    media = Media(video, tmp_path, settings)
    with pytest.raises(ValueError, match="至少需要 2 秒"):
        asyncio.run(media.clip(0, 1.5))


def test_clip_rejects_oversized_range(tmp_path: Path, video, settings):
    media = Media(video, tmp_path, settings.model_copy(update={"max_video_seconds": 2}))
    with pytest.raises(ValueError, match="最长 2 秒"):
        asyncio.run(media.clip(0, 3))


def test_crop_is_applied_before_scaling(tmp_path: Path, video, settings):
    media = Media(video, tmp_path, settings)
    asyncio.run(media.clip(0, 2, crop=[0, 0, 0.5, 0.5]))
    encoded = asyncio.run(probe(media.clips / "clip.mp4"))
    assert (encoded.width, encoded.height) == (160, 120)
