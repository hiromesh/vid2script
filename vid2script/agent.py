import json
import logging
from pathlib import Path

from pydantic import Field, ValidationError

from .asr import Transcript
from .llm import History, InputTooLarge, Model, OutputTruncated
from .media import Media
from .script import Patch, Record, Script, dump

log = logging.getLogger(__name__)


class Inspect(Record):
    start: float = Field(ge=0)
    end: float = Field(gt=0)
    question: str = Field(min_length=1, max_length=300)
    crop: list[float] | None = Field(default=None, min_length=4, max_length=4)


class Read(Record):
    scene_id: str = ""
    offset: int = Field(default=0, ge=0)
    limit: int = Field(default=10, ge=1, le=12)
    character: str = ""
    previous_episode: str = ""


TOOLS = [
    {"type": "function", "function": {"name": name, "description": description, "parameters": schema.model_json_schema()}}
    for name, description, schema in (
        ("inspect_clip", "重新观察已经提供过的时间范围，核对人物、动作或字幕；可裁剪局部。", Inspect),
        ("read_script", "分页读取已有场景正文或人物及场景索引。previous_episode 可读取已完成集数的 Markdown，offset 为行号。", Read),
        ("update_script", "原子提交有限的剧本增量或局部修订；每次一个场景，最多 12 条正文。", Patch),
    )
]


class Agent:
    def __init__(self, model: Model):
        self.model = model
        self.history = History(model)
        self.previous: dict[str, Path] = {}
        self.characters = {}
        self.continuity = ""

    async def episode(self, media: Media, transcript: Transcript, script: Script,
                      output: Path, window: tuple[float, float]) -> None:
        script.characters = dict(self.characters)
        script.continuity = self.continuity
        cursor, stop = window
        try:
            while cursor < stop - 1e-6:
                end = min(stop, cursor + self.model.settings.batch_seconds)
                end = await self._batch(media, transcript, script, (cursor, end))
                cursor = end
                script.save(output, False, media.video.path, window)
                log.info("%s %.1f/%.1fs scenes=%s blocks=%s calls=%s compact=%s",
                         script.episode, cursor, stop, len(script.scenes), script.block_counter,
                         self.model.calls, self.model.compactions)
            await self._review(media, transcript, script, window)
            self._validate_completion(script, window)
        except BaseException:
            if script.scenes or script.coverage:
                script.save(output, False, media.video.path, window)
            raise
        script.save(output, True, media.video.path, window)
        self.previous[script.episode] = output
        self.characters, self.continuity = dict(script.characters), script.continuity

    async def _batch(self, media: Media, transcript: Transcript, script: Script,
                     window: tuple[float, float]) -> float:
        start, end = window
        end = min(end, start + self.model.settings.max_images * media.video.interval)
        while True:
            await transcript.ensure(min(media.video.duration, end + 3))
            script.cues.update({c.id: c for c in transcript.cues})
            # 每次只发送当前片段图片；前段衔接由精确保留的正文和连续性状态提供。
            content, frames = await media.images(start, end)
            cues = transcript.between(max(0, start - 3), end + 3)
            used = script.used_cues()
            intro = {"episode": script.episode, "title": script.title, "duration": media.video.duration,
                     "responsible_range": [start, end], "sampling_interval": media.video.interval,
                     "transcript": [{**c.model_dump(), "already_written": c.id in used} for c in cues], "state": script.state(),
                     "missing_cues": script.missing(start, end), "previous_episodes": list(self.previous),
                     "instruction": "观察当前片段并增量提交剧本；参考范围中的旧台词不得重复。"}
            message = [{"type": "text", "text": dump(intro)}, *content]
            predicted = self.model.estimate([self.history.messages[0], {"role": "user", "content": message}], TOOLS)
            if (predicted < self.model.settings.context_window_tokens * 0.55
                    and len(dump(message).encode()) < self.model.settings.max_request_bytes * 0.8):
                break
            if end - start <= media.video.interval * 2:
                raise InputTooLarge("最小片段仍然超过预算，请增加上下文容量或减少图片尺寸")
            end = start + max(media.video.interval, int((end - start) / 2 / media.video.interval) * media.video.interval)
        script.frame_ids.update(frames)
        self.history.begin(message)
        await self._loop(media, script, (start, end), review=False)
        return end

    async def _loop(self, media: Media, script: Script, window: tuple[float, float], review: bool) -> None:
        inspections = truncations = 0
        for _ in range(self.model.settings.max_turns_per_batch):
            await self.history.prepare(TOOLS, script.state())
            try:
                response = await self.model.call(self.history.messages, TOOLS)
            except OutputTruncated:
                truncations += 1
                if truncations > 3:
                    raise
                self.history.messages.append({"role": "user", "content":
                    f"刚才输出超过单次上限，没有提交。每次最多提交 {max(1, 6 // truncations)} 条短正文，分多次调用；不要重写已提交内容。"})
                continue
            if not response.tool_calls:
                self.history.assistant(response)
                self.history.messages.append({"role": "user", "content":
                    "本段尚未提交完成。请调用 update_script 分段提交，全部写完后设置 batch_complete=true。"})
                continue
            self.history.assistant(response)
            completed = False
            pending_images = []
            for call in response.tool_calls:
                try:
                    result, images = await self._execute(call, media, script, window, review, inspections)
                    if call.function.name == "inspect_clip":
                        inspections += 1
                    pending_images.extend(images)
                    completed = completed or result.get("batch_complete", False)
                except (ValidationError, ValueError, json.JSONDecodeError) as error:
                    detail = str(error)[:1800]
                    if isinstance(error, json.JSONDecodeError):
                        detail += f"；错误位置附近：{error.doc[max(0, error.pos - 80):error.pos + 80]}"
                    result = {"accepted": False, "error": detail,
                              "transaction": "本次操作全部未写入；此前成功操作保留", "state": script.state(),
                              "missing_cues": script.missing(*window)[:20]}
                    log.info("工具参数校验：%s", detail[:400])
                self.history.result(call.id, result)
            if pending_images:
                # 新的观察替换当前旧图片，历史仍保留时间戳和工具结果。
                for message in self.history.messages:
                    if isinstance(message.get("content"), list):
                        message["content"] = [p for p in message["content"] if p.get("type") != "image_url"]
                self.history.messages.append({"role": "user", "content": pending_images})
            if completed:
                return
        raise RuntimeError(f"{window} 超过单段工具调用上限；已写正文保留，未标记完成")

    async def _execute(self, call, media: Media, script: Script, window: tuple[float, float],
                       review: bool, inspections: int) -> tuple[dict, list[dict]]:
        name, raw = call.function.name, call.function.arguments
        arguments = json.loads(raw)
        if name == "update_script":
            patch = Patch.model_validate(arguments)
            return script.apply(patch, window, review), []
        if name == "read_script":
            args = Read.model_validate(arguments)
            if args.previous_episode:
                if args.previous_episode not in self.previous:
                    raise ValueError("指定集数尚未完成")
                lines = self.previous[args.previous_episode].read_text().splitlines()
                return {"total_lines": len(lines), "lines": lines[args.offset:args.offset + args.limit * 3]}, []
            return script.read(args.scene_id, args.offset, args.limit, args.character), []
        if name == "inspect_clip":
            if inspections >= 3:
                raise ValueError("本段补充观察已达 3 次，请根据已有证据写作，将未解决问题写入 issues")
            args = Inspect.model_validate(arguments)
            observed_end = max(window[1], max((c["end"] for c in script.coverage), default=0))
            if args.end <= args.start or args.end > observed_end + 0.001:
                raise ValueError("观察范围必须在已经处理的时间范围内")
            images, frames = await media.images(args.start, args.end, args.crop)
            script.frame_ids.update(frames)
            return {"frame_ids": frames, "question": args.question}, [
                {"type": "text", "text": f"补充观察：{args.question}"}, *images]
        raise ValueError(f"未知工具：{name}")

    async def _review(self, media: Media, transcript: Transcript, script: Script,
                      window: tuple[float, float]) -> None:
        log.info("开始逐页核对 %s", script.episode)
        for page in script.pages():
            log.info("核对场景 %s，正文 %s–%s", page["scene_id"], page["blocks"][0]["id"], page["blocks"][-1]["id"])
            ids = {cue for block in page["blocks"] for cue in block["cue_ids"]}
            prompt = {"mode": "review", "page": page, "state": script.state(),
                      "source_dialogue": [script.cues[key].model_dump() for key in sorted(ids)],
                      "instruction": "核对本页人物、台词、动作承接与重复。仅提交必要局部修订；完成后 batch_complete=true。"}
            times = sorted({int(frame[1:]) / 1000 for block in page["blocks"] for frame in block["frame_ids"]})
            evidence = []
            if times:
                selected = sorted({times[0], times[len(times) // 2], times[-1]})
                for timestamp in selected:
                    images, _ = await media.images(timestamp, min(media.video.duration, timestamp + media.video.interval))
                    evidence.extend(images)
            self.history.begin([{"type": "text", "text": dump(prompt)}, *evidence])
            await self._loop(media, script, window, review=True)
            log.info("本页核对完成")
        if not script.scenes:
            log.info("当前范围没有剧情正文，保留非正文片段说明")

    @staticmethod
    def _validate_completion(script: Script, window: tuple[float, float]) -> None:
        cursor = window[0]
        for item in script.coverage:
            if abs(item["start"] - cursor) > 0.001:
                raise ValueError("视频处理范围存在遗漏或重叠")
            cursor = item["end"]
        if abs(cursor - window[1]) > 0.001 or script.missing(*window):
            raise ValueError("仍有未处理时间或台词，不能标记剧本已完成")
