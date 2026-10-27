import asyncio
import json
import logging
import time
from pathlib import Path
from urllib.parse import urlsplit
from uuid import uuid4

import httpx
import oss2

from .config import Settings
from .media import Cue, Media

log = logging.getLogger(__name__)
SUBMIT_URL = "https://openspeech.bytedance.com/api/v3/auc/bigmodel/submit"
QUERY_URL = "https://openspeech.bytedance.com/api/v3/auc/bigmodel/query"


class Speech:
    def __init__(self, settings: Settings):
        settings.require_asr()
        self.settings = settings
        endpoint = settings.oss_endpoint
        parsed = urlsplit(endpoint if "://" in endpoint else f"https://{endpoint}")
        if not parsed.hostname:
            raise ValueError("OSS endpoint 无效")
        self.bucket = oss2.Bucket(
            oss2.Auth(settings.oss_access_key_id.get_secret_value(), settings.oss_access_key_secret.get_secret_value()),
            f"https://{parsed.netloc}", settings.oss_bucket, connect_timeout=60,
        )
        self.active_keys: set[str] = set()

    async def transcribe(self, path: Path, segment: int, window: tuple[float, float, float]) -> list[Cue]:
        offset, core_start, core_end = window
        key = f"vid2script/audio/{uuid4().hex}.wav"
        self.active_keys.add(key)
        try:
            upload = asyncio.create_task(asyncio.to_thread(self.bucket.put_object_from_file, key, str(path),
                                                          headers={"Content-Type": "audio/wav"}))
            try:
                await asyncio.shield(upload)
            except asyncio.CancelledError:
                # 等上传实际结束再删除，避免取消时出现“先删除、后上传完成”的残留。
                await upload
                raise
            url = self.bucket.sign_url("GET", key, int(self.settings.asr_timeout_seconds + 600), slash_safe=True)
            body = await self.recognize(url)
            return parse_transcript(body, segment, offset, core_start, core_end)
        finally:
            try:
                await asyncio.to_thread(self.bucket.delete_object, key)
            except Exception:
                log.warning("删除 OSS 临时对象失败：%s", key, exc_info=True)
            self.active_keys.discard(key)
            path.unlink(missing_ok=True)

    async def recognize(self, audio_url: str) -> dict:
        request_id = str(uuid4())
        headers = {"Content-Type": "application/json", "X-Api-Key": self.settings.asr_api_key.get_secret_value(),
                   "X-Api-Resource-Id": self.settings.asr_resource_id, "X-Api-Request-Id": request_id}
        payload = {"user": {"uid": "vid2script"}, "audio": {"format": "wav", "url": audio_url},
                   "request": {"model_name": "bigmodel", "enable_itn": True, "enable_punc": True,
                               "show_utterances": True, "vad_segment": True, "end_window_size": 500,
                               "enable_speaker_info": True, "ssd_version": "200"}}
        deadline = time.monotonic() + self.settings.asr_timeout_seconds
        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(SUBMIT_URL, headers={**headers, "X-Api-Sequence": "-1"}, json=payload)
            check_status(response, {20000000}, request_id)
            while time.monotonic() < deadline:
                await asyncio.sleep(self.settings.asr_poll_interval_seconds)
                response = await client.post(QUERY_URL, headers=headers, json={})
                status = check_status(response, {20000000, 20000001, 20000002, 20000003}, request_id)
                if status == 20000003:
                    return {"result": {"utterances": []}}
                if status == 20000000:
                    return response.json()
        raise TimeoutError(f"字节 ASR 任务超时：{request_id}")


def check_status(response: httpx.Response, accepted: set[int], request_id: str) -> int:
    response.raise_for_status()
    status = int(response.headers["X-Api-Status-Code"])
    if status not in accepted:
        raise RuntimeError(f"字节 ASR 失败：status={status}, request={request_id}, "
                           f"message={response.headers.get('X-Api-Message', '')}")
    return status


def parse_transcript(body: dict, segment: int, offset: float, start: float, end: float) -> list[Cue]:
    result = body.get("result", {})
    utterances = result.get("utterances")
    if not isinstance(utterances, list):
        raise TypeError("ASR 没有返回带时间戳的 utterances，无法可靠关联台词")
    cues = []
    for index, item in enumerate(split_utterances(utterances)):
        left, right = offset + item["start_time"] / 1000, offset + item["end_time"] / 1000
        text = item.get("text", "").strip()
        # 分段重叠部分使用逐词时间戳确定归属，避免重复整句。
        if left < start or right > end:
            words = [w for w in item.get("words", [])
                     if start <= offset + (w["start_time"] + w["end_time"]) / 2000 < end]
            if words:
                text = "".join(w["text"] for w in words)
                left, right = offset + words[0]["start_time"] / 1000, offset + words[-1]["end_time"] / 1000
            elif not start <= (left + right) / 2 < end:
                continue
        if not text:
            continue
        additions = item.get("additions") or {}
        if isinstance(additions, str):
            additions = json.loads(additions)
        speaker = item.get("speaker_id", item.get("speaker", additions.get("speaker", "")))
        cues.append(Cue(id=f"asr_{segment:04}_{index:05}", start=max(0, left), end=right, text=text,
                        speaker=f"audio_{segment}:speaker_{speaker}" if speaker != "" else ""))
    return cues


def split_utterances(utterances: list[dict]) -> list[dict]:
    output = []
    for item in utterances:
        words = item.get("words") or []
        if not words:
            output.append(item)
            continue
        groups: list[list[dict]] = []
        for word in words:
            if not groups or (word["start_time"] - groups[-1][-1]["end_time"] >= 600
                              or word["end_time"] - groups[-1][0]["start_time"] > 6000
                              or groups[-1][-1]["text"].endswith(("。", "！", "？", "!", "?"))):
                groups.append([])
            groups[-1].append(word)
        for group in groups:
            output.append({**item, "start_time": group[0]["start_time"], "end_time": group[-1]["end_time"],
                           "text": "".join(word["text"] for word in group), "words": group})
    return output


class Transcript:
    def __init__(self, media: Media, speech: Speech | None, subtitles: list[Cue] | None, stop: float | None = None):
        self.media, self.speech = media, speech
        self.stop = stop if stop is not None else media.video.duration
        self.cues = subtitles if subtitles is not None else []
        self.ready_until = media.video.duration if subtitles is not None or not media.video.audio else 0.0
        self.segment = 0

    async def ensure(self, end: float) -> None:
        end = min(end, self.stop)
        while self.ready_until < end - 1e-6:
            if self.speech is None:
                raise ValueError("视频包含音频，但 ASR 未配置")
            start = self.ready_until
            target = min(self.stop, start + self.media.settings.asr_segment_seconds)
            stop = min(self.stop, await self.media.silence_cut(target)) if target < self.stop else target
            offset = max(0, start - 2)
            path = await self.media.audio(offset, min(self.stop, stop + 2))
            log.info("ASR %.1f–%.1fs", start, stop)
            self.cues.extend(await self.speech.transcribe(path, self.segment, (offset, start, stop)))
            self.ready_until, self.segment = stop, self.segment + 1

    def between(self, start: float, end: float) -> list[Cue]:
        return [cue for cue in self.cues if cue.end > start and cue.start < end]
