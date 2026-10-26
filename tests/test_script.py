from pathlib import Path

import pytest

from vid2script.media import Cue
from vid2script.script import Block, Character, Patch, Replacement, SceneInfo, Script


def initial() -> Script:
    script = Script("测试剧", "第01集")
    script.frame_ids.add("f0")
    script.cues["c0"] = Cue(id="c0", start=1, end=2, text="你回来了。")
    return script


def first_patch(**changes) -> Patch:
    values = {
        "operation_id": "op1", "scene_id": "s1",
        "scene": SceneInfo(location="客厅", setting="内景", time_of_day="夜"),
        "characters": [Character(id="p1", name="男子")],
        "append_blocks": [Block(kind="dialogue", text="你回来了。", character_id="p1", start=1, end=2, cue_ids=["c0"])],
        "batch_complete": True,
    }
    return Patch(**{**values, **changes})


def test_atomic_rejection_preserves_everything():
    script = initial()
    bad = first_patch(append_blocks=[Block(kind="dialogue", text="你回来了。", character_id="missing",
                                          start=1, end=2, cue_ids=["c0"])])
    with pytest.raises(ValueError, match="人物不存在"):
        script.apply(bad, (0, 30))
    assert not script.scenes and not script.characters and not script.operations and not script.coverage
    assert script.block_counter == 0


def test_operation_retry_does_not_duplicate_scene_or_coverage():
    script = initial()
    patch = first_patch()
    result = script.apply(patch, (0, 30))
    assert script.apply(patch, (0, 30)) == result
    assert len(script.scenes["s1"].blocks) == len(script.coverage) == 1
    with pytest.raises(ValueError, match="operation_id"):
        script.apply(first_patch(continuity="不同内容"), (0, 30))


def test_missing_dialogue_prevents_progress():
    script = initial()
    script.cues["c1"] = Cue(id="c1", start=5, end=6, text="请进。")
    with pytest.raises(ValueError, match="未写入"):
        script.apply(first_patch(), (0, 30))
    assert script.block_counter == 0


def test_duplicate_dialogue_rejected_even_with_new_operation_id():
    script = initial()
    script.apply(first_patch(), (0, 30))
    with pytest.raises(ValueError, match="重复"):
        script.apply(first_patch(operation_id="op2", batch_complete=False), (0, 30))
    assert script.block_counter == 1


def test_replacement_keeps_order_and_character_rename_renders(tmp_path: Path):
    script = initial()
    result = script.apply(first_patch(), (0, 30))
    replacement = Block(kind="dialogue", text="你回来了？", character_id="p1", start=1, end=2, cue_ids=["c0"])
    script.apply(Patch(operation_id="op2", scene_id="s1", characters=[Character(id="p1", name="陈默")],
                       replacements=[Replacement(block_id=result["block_ids"][0], block=replacement)]), (0, 30), True)
    target = tmp_path / "episode.md"
    script.save(target, True, Path("source.mp4"), (0, 30))
    text = target.read_text()
    assert "**陈默**：你回来了？" in text and "**男子**" not in text
    assert script.block_counter == 1


def test_long_scene_is_bounded_per_call_and_complete_on_disk(tmp_path: Path):
    script = initial()
    script.apply(first_patch(), (0, 30))
    for index in range(1, 361):
        script.apply(Patch(operation_id=f"op{index + 1}", scene_id="s1", append_blocks=[
            Block(kind="action", text=f"动作 {index}。", start=index * 30, end=index * 30 + 2, frame_ids=["f0"])
        ], batch_complete=True), (index * 30, (index + 1) * 30))
    assert len(script.scenes) == 1
    assert len(script.scenes["s1"].blocks) == 361
    assert len(script.read("s1", 350, 10)["blocks"]) == 10
    target = tmp_path / "movie.md"
    script.save(target, True, Path("movie.mp4"), (0, 10830))
    assert "动作 1。" in target.read_text() and "动作 360。" in target.read_text()


def test_invalid_patch_json_cannot_mutate_script():
    script = initial()
    with pytest.raises(ValueError):
        Patch.model_validate_json('{"operation_id":"op1","append_blocks":[')
    assert not script.scenes


def test_deleting_only_dialogue_cannot_complete_batch():
    script = initial()
    result = script.apply(first_patch(batch_complete=False), (0, 30))
    with pytest.raises(ValueError, match="未写入"):
        script.apply(Patch(operation_id="op2", scene_id="s1", delete_block_ids=result["block_ids"],
                           batch_complete=True), (0, 30))
    assert len(script.scenes["s1"].blocks) == 1


def test_review_can_delete_incorrect_action_without_losing_dialogue():
    script = initial()
    script.apply(first_patch(), (0, 30))
    result = script.apply(Patch(operation_id="op2", scene_id="s1", append_blocks=[
        Block(kind="action", text="推测性动作。", start=3, end=4, frame_ids=["f0"])
    ]), (0, 30))
    script.apply(Patch(operation_id="op3", scene_id="s1", delete_block_ids=result["block_ids"],
                       batch_complete=True), (0, 30), True)
    assert len(script.scenes["s1"].blocks) == 1 and not script.missing(0, 30)


def test_review_can_correct_an_earlier_block_during_current_batch():
    script = initial()
    result = script.apply(first_patch(), (0, 30))
    corrected = Block(kind="dialogue", text="你回来了？", character_id="p1", start=1, end=2, cue_ids=["c0"])
    script.apply(Patch(operation_id="op2", scene_id="s1", replacements=[
        Replacement(block_id=result["block_ids"][0], block=corrected)
    ]), (90, 120))
    assert next(iter(script.scenes["s1"].blocks.values())).text == "你回来了？"
