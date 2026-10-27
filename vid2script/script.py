import hashlib
import json
import unicodedata
from collections.abc import Iterator
from copy import deepcopy
from pathlib import Path
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .media import Cue


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Character(Record):
    id: str = Field(min_length=1, max_length=50)
    name: str = Field(min_length=1, max_length=40)
    description: str = Field(default="", max_length=200)
    aliases: list[str] = Field(default_factory=list, max_length=10)


class Block(Record):
    kind: Literal["action", "dialogue", "sound"]
    text: str = Field(min_length=1, max_length=1000)
    character_id: str = ""
    direction: str = Field(default="", max_length=120)
    delivery: Literal["", "画外", "电话", "旁白", "内心独白"] = ""
    start: float = Field(ge=0)
    end: float = Field(ge=0)
    cue_ids: list[str] = Field(default_factory=list, max_length=20)
    video_ids: list[str] = Field(default_factory=list, max_length=10)

    @model_validator(mode="after")
    def valid(self) -> "Block":
        if self.end < self.start:
            raise ValueError("正文结束时间早于开始时间")
        if self.kind == "dialogue" and not self.character_id:
            raise ValueError("对白必须关联人物；身份不明时建立暂用称呼")
        if not self.cue_ids and not self.video_ids:
            raise ValueError("正文必须引用台词或视频片段证据")
        return self


class SceneInfo(Record):
    location: str = Field(min_length=1, max_length=100)
    setting: Literal["内景", "外景", "内外景", "待确认"]
    time_of_day: str = Field(min_length=1, max_length=40)
    narrative: str = Field(default="", max_length=40)
    character_ids: list[str] = Field(default_factory=list, max_length=30)


class Replacement(Record):
    block_id: str
    block: Block


class Exclusion(Record):
    cue_id: str
    reason: Literal["歌词", "片头片尾", "前情回顾", "重复识别", "识别错误", "非剧情内容"]
    detail: str = Field(default="", max_length=160)


class Patch(Record):
    operation_id: str = Field(min_length=1, max_length=80)
    scene_id: str = Field(default="", max_length=50)
    scene: SceneInfo | None = None
    characters: list[Character] = Field(default_factory=list, max_length=8)
    append_blocks: list[Block] = Field(default_factory=list, max_length=12)
    replacements: list[Replacement] = Field(default_factory=list, max_length=12)
    delete_block_ids: list[str] = Field(default_factory=list, max_length=12)
    excluded_cues: list[Exclusion] = Field(default_factory=list, max_length=32)
    issues: list[str] = Field(default_factory=list, max_length=10)
    scene_finished: bool = False
    continuity: str = Field(default="", max_length=1600)
    batch_complete: bool = False
    skip_reason: str = Field(default="", max_length=160)


class Scene(Record):
    id: str
    info: SceneInfo
    blocks: dict[str, Block] = Field(default_factory=dict)
    finished: bool = False


class Script:
    def __init__(self, title: str, episode: str):
        self.title, self.episode = title, episode
        self.characters: dict[str, Character] = {}
        self.scenes: dict[str, Scene] = {}
        self.operations: dict[str, tuple[str, dict]] = {}
        self.cues: dict[str, Cue] = {}
        self.video_ids: set[str] = set()
        self.exclusions: dict[str, Exclusion] = {}
        self.issues: list[str] = []
        self.coverage: list[dict] = []
        self.continuity = ""
        self.block_counter = 0

    def used_cues(self, scenes: dict[str, Scene] | None = None) -> set[str]:
        return {cue for scene in (scenes or self.scenes).values() for block in scene.blocks.values()
                for cue in block.cue_ids}

    def missing(self, start: float, end: float) -> list[str]:
        used = self.used_cues() | self.exclusions.keys()
        return [cue.id for cue in self.cues.values() if start <= cue.start < end and cue.id not in used]

    def apply(self, patch: Patch, window: tuple[float, float], review: bool = False) -> dict:
        digest = hashlib.sha256(patch.model_dump_json().encode()).hexdigest()
        if patch.operation_id in self.operations:
            previous, result = self.operations[patch.operation_id]
            if previous != digest:
                raise ValueError("operation_id 已用于其他内容，请使用新的 ID")
            return result
        chars = {**self.characters, **{c.id: c for c in patch.characters}}
        scenes = dict(self.scenes)
        scene = self._scene(patch, scenes)
        if review and patch.append_blocks:
            raise ValueError("核对阶段使用 replacements 修订或 delete_block_ids 删除，不追加未经核对的正文")
        next_id = self.block_counter
        created = []
        if scene:
            if any(key not in chars for key in scene.info.character_ids):
                raise ValueError("场景的出场人物尚未登记")
            for block_id in patch.delete_block_ids:
                if block_id not in scene.blocks:
                    raise ValueError(f"要删除的正文条目不存在：{block_id}")
                del scene.blocks[block_id]
            for replacement in patch.replacements:
                if replacement.block_id not in scene.blocks:
                    raise ValueError(f"正文条目不存在：{replacement.block_id}")
                self._validate_block(replacement.block, chars, window, True)
                scene.blocks[replacement.block_id] = replacement.block
            for block in patch.append_blocks:
                self._validate_block(block, chars, window, review)
                next_id += 1
                block_id = f"b{next_id:07}"
                created.append(block_id)
                scene.blocks[block_id] = block
            scene.finished = patch.scene_finished
            scenes[scene.id] = scene
        excluded = self._excluded(patch, scenes)
        used = self.used_cues(scenes)
        self._validate_duplicates(scenes)
        missing = [c.id for c in self.cues.values() if window[0] <= c.start < window[1]
                   and c.id not in used and c.id not in excluded]
        if patch.batch_complete and missing and not review:
            raise ValueError(f"本段还有台词未写入或说明排除原因：{missing[:20]}")
        if patch.batch_complete and not review:
            has_blocks = any(window[0] <= b.start < window[1] for s in scenes.values() for b in s.blocks.values())
            if not has_blocks and not patch.skip_reason:
                raise ValueError("本段没有正文，请说明跳过原因")
        # 所有校验完成后一次提交，错误和截断不能留下半份正文。
        self.characters, self.scenes, self.exclusions = chars, scenes, excluded
        self.block_counter = next_id
        self.continuity = patch.continuity or self.continuity
        self.issues.extend(f"{clock(window[0])}：{item}" for item in patch.issues if item)
        if patch.batch_complete and not review:
            self.coverage.append({"start": window[0], "end": window[1], "skip_reason": patch.skip_reason})
        result = {"accepted": True, "scene_id": patch.scene_id, "block_ids": created,
                  "batch_complete": patch.batch_complete, "remaining_cues": missing[:20]}
        self.operations[patch.operation_id] = (digest, result)
        return result

    def _scene(self, patch: Patch, scenes: dict[str, Scene]) -> Scene | None:
        if not patch.scene_id:
            if patch.append_blocks or patch.replacements or patch.delete_block_ids or patch.scene:
                raise ValueError("修改场景正文需要 scene_id")
            return None
        if patch.scene_id in scenes:
            scene = deepcopy(scenes[patch.scene_id])
            if patch.scene:
                scene.info = patch.scene
            return scene
        if patch.scene is None:
            raise ValueError("新场景必须提供 scene 元信息")
        return Scene(id=patch.scene_id, info=patch.scene)

    def _validate_block(self, block: Block, chars: dict, window: tuple[float, float], review: bool) -> None:
        if block.character_id and block.character_id not in chars:
            raise ValueError(f"人物不存在：{block.character_id}")
        if any(c not in self.cues for c in block.cue_ids):
            raise ValueError("正文引用了未提供的台词 ID")
        if any(v not in self.video_ids for v in block.video_ids):
            raise ValueError("正文引用了未观察的视频片段 ID")
        if not review and (block.start < window[0] - 3 or block.start >= window[1] + 0.001):
            raise ValueError(f"新增正文 start={block.start} 超出当前负责范围 {window}：{block.text[:60]}。"
                             "此前正文应使用 replacements 和原 block_id 修订，不能再次追加。")
        evidence_ends = [self.cues[c].end for c in block.cue_ids]
        if block.end > max([window[1] + 3, *evidence_ends]) + 0.001:
            raise ValueError("正文结束时间超出已提供的证据范围")

    def _excluded(self, patch: Patch, scenes: dict[str, Scene]) -> dict[str, Exclusion]:
        excluded = dict(self.exclusions)
        used = self.used_cues(scenes)
        for cue in used:
            excluded.pop(cue, None)
        for item in patch.excluded_cues:
            if item.cue_id not in self.cues or item.cue_id in used:
                raise ValueError("排除的台词不存在或已经用于正文")
            excluded[item.cue_id] = item
        return excluded

    @staticmethod
    def _validate_duplicates(scenes: dict[str, Scene]) -> None:
        seen = set()
        previous: list[Block] = []
        for scene in scenes.values():
            for block in scene.blocks.values():
                if block.kind != "dialogue":
                    continue
                identity = (tuple(sorted(block.cue_ids or block.video_ids)), block.character_id, block.text)
                if identity in seen:
                    raise ValueError("同一台词被重复写入，请修改已有条目")
                seen.add(identity)
                text = normalized(block.text)
                for other in previous[-20:]:
                    if not set(other.cue_ids).intersection(block.cue_ids):
                        continue
                    older = normalized(other.text)
                    if min(len(text), len(older)) >= 6 and (text in older or older in text):
                        raise ValueError("同一台词证据的内容重复，请读取并修订已有条目")
                previous.append(block)

    def read(self, scene_id: str = "", offset: int = 0, limit: int = 10, character: str = "") -> dict:
        if scene_id:
            if scene_id not in self.scenes:
                raise ValueError("场景不存在")
            scene = self.scenes[scene_id]
            blocks = list(scene.blocks.items())
            return {"scene_id": scene.id, "scene": scene.info.model_dump(), "total": len(blocks),
                    "next_offset": min(len(blocks), offset + limit),
                    "blocks": [{"id": k, **v.model_dump()} for k, v in blocks[offset:offset + limit]],
                    "characters": [self.characters[key].model_dump() for key in dict.fromkeys(
                        b.character_id for _, b in blocks[offset:offset + limit] if b.character_id)]}
        scenes = [s for s in self.scenes.values() if not character or
                  any(b.character_id == character for b in s.blocks.values())]
        return {"total": len(scenes), "next_offset": min(len(scenes), offset + limit),
                "scenes": [self._index(s) for s in scenes[offset:offset + limit]],
                "characters": [c.model_dump() for c in list(self.characters.values())[offset:offset + limit]]}

    @staticmethod
    def _index(scene: Scene) -> dict:
        return {"id": scene.id, **scene.info.model_dump(), "blocks": len(scene.blocks), "finished": scene.finished}

    def state(self) -> dict:
        recent = list(self.scenes.values())[-3:]
        ids = {b.character_id for s in recent for b in s.blocks.values() if b.character_id}
        return {"continuity": self.continuity,
                "recent_scenes": [self._index(s) for s in recent],
                "recent_blocks": [{"id": k, **v.model_dump()} for s in recent[-1:]
                                  for k, v in list(s.blocks.items())[-4:]],
                "characters": [self.characters[key].model_dump() for key in sorted(ids)][:20]}

    def pages(self) -> Iterator[dict]:
        for scene_id in list(self.scenes):
            block_ids = list(self.scenes[scene_id].blocks)
            for offset in range(0, len(block_ids), 10):
                scene = self.scenes[scene_id]
                blocks = [{"id": key, **scene.blocks[key].model_dump()} for key in block_ids[offset:offset + 10]
                          if key in scene.blocks]
                if blocks:
                    yield {"scene_id": scene.id, "scene": scene.info.model_dump(), "blocks": blocks,
                           "characters": [c.model_dump() for c in self.characters.values()
                                          if c.id in scene.info.character_ids
                                          or any(b["character_id"] == c.id for b in blocks)]}

    def save(self, target: Path, complete: bool, source: Path, window: tuple[float, float]) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        lines = [f"# {self.title} {self.episode}", "", f"来源：{source.name}",
                 "输入：视频片段（video_url）",
                 f"处理范围：{clock(window[0])}–{clock(window[1])}",
                 f"状态：{'已完成' if complete else '处理中，尚未完成核对'}", "", "## 本集人物", ""]
        used = {b.character_id for scene in self.scenes.values() for b in scene.blocks.values() if b.character_id}
        used.update(key for scene in self.scenes.values() for key in scene.info.character_ids)
        lines.extend(f"- **{c.name}**：{c.description or '身份待确认'}"
                     for c in self.characters.values() if c.id in used)
        for number, scene in enumerate(self.scenes.values(), 1):
            lines.extend(self._render_scene(scene, number))
        notes = list(self.issues)
        notes.extend(f"{clock(self.cues[e.cue_id].start)}：{e.reason}；{e.detail}"
                     for e in self.exclusions.values() if e.reason == "识别错误")
        if notes:
            lines.extend(["", "<details>", "<summary>待核对内容</summary>", "",
                          *[f"- {note}" for note in dict.fromkeys(notes)], "", "</details>"])
        skipped = [c for c in self.coverage if c["skip_reason"]]
        if skipped:
            lines.extend(["", "<details>", "<summary>非正文片段</summary>", ""])
            lines.extend(f"- {clock(c['start'])}–{clock(c['end'])}：{c['skip_reason']}" for c in skipped)
            lines.extend(["", "</details>"])
        temporary = target.with_name(f".{target.name}.{uuid4().hex}.part")
        try:
            temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)

    def _render_scene(self, scene: Scene, number: int) -> list[str]:
        info = scene.info
        lines = ["", f"## {number:02} {info.location} · {info.setting} · {info.time_of_day}"
                 + (f" · {info.narrative}" if info.narrative else ""), ""]
        actors = dict.fromkeys([*scene.info.character_ids,
                               *(b.character_id for b in scene.blocks.values() if b.character_id)])
        if actors:
            lines.extend(["人物：" + "、".join(self.characters[c].name for c in actors), ""])
        for block in scene.blocks.values():
            text = block.text
            if block.kind == "dialogue":
                hints = "，".join(x for x in (block.delivery, block.direction) if x)
                text = f"**{self.characters[block.character_id].name}**" + (f"（{hints}）" if hints else "") + f"：{text}"
            lines.extend([text, ""])
        if scene.blocks:
            start = min(b.start for b in scene.blocks.values())
            end = max(b.end for b in scene.blocks.values())
            lines.extend(["<details>", "<summary>原片位置</summary>", "", f"{clock(start)}–{clock(end)}", "", "</details>"])
        return lines


def clock(seconds: float) -> str:
    value = int(seconds)
    return f"{value // 3600:02}:{value // 60 % 60:02}:{value % 60:02}"


def dump(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def normalized(text: str) -> str:
    return "".join(c for c in text if not c.isspace() and not unicodedata.category(c).startswith("P"))
