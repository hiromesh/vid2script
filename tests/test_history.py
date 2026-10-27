from vid2script.llm import compact_parts, text_only


def test_compact_boundary_preserves_last_tool_exchange_and_active_task():
    messages = [
        {"role": "system", "content": "规则"},
        {"role": "user", "content": "上一段"},
        {"role": "assistant", "tool_calls": [{"id": "old"}]},
        {"role": "tool", "tool_call_id": "old", "content": "已写入"},
        {"role": "user", "content": [{"type": "text", "text": "当前片段与台词"},
                                      {"type": "video_url", "video_url": {"url": "data:video/mp4;base64,AAAA"}}]},
        {"role": "assistant", "tool_calls": [{"id": "first"}]},
        {"role": "tool", "tool_call_id": "first", "content": "已写入"},
        {"role": "assistant", "tool_calls": [{"id": "last"}]},
        {"role": "tool", "tool_call_id": "last", "content": "请修正参数"},
    ]
    old, kept = compact_parts(messages, 4)
    assert [m["tool_call_id"] for m in old if m["role"] == "tool"] == ["old", "first"]
    assert [m["tool_call_id"] for m in kept if m["role"] == "tool"] == ["last"]
    assert kept[0]["content"] == [{"type": "text", "text": "当前片段与台词"}]
    assert kept[1]["tool_calls"][0]["id"] == "last"


def test_text_only_removes_video_without_mutating_original():
    messages = [
        {"role": "user", "content": [{"type": "text", "text": "f1"},
                                      {"type": "video_url", "video_url": {"url": "data:video/mp4;base64,AAAA"}}]},
        {"role": "assistant", "reasoning_content": "推理", "tool_calls": [{"id": "call1"}]},
        {"role": "tool", "tool_call_id": "call1", "content": "已提交"},
    ]
    kept = text_only(messages)
    assert kept[0]["content"] == [{"type": "text", "text": "f1"}]
    assert kept[1:] == messages[1:]
    assert len(messages[0]["content"]) == 2
    assert "reasoning_content" not in text_only(messages, keep_reasoning=False)[1]
    assert "reasoning_content" in messages[1]


def test_compact_without_assistant_keeps_current_video():
    video = {"type": "video_url", "video_url": {"url": "data:video/mp4;base64,AAAA"}}
    messages = [
        {"role": "system", "content": "规则"},
        {"role": "user", "content": "上一段"},
        {"role": "user", "content": [{"type": "text", "text": "当前片段"}, video]},
    ]
    old, kept = compact_parts(messages, 2)
    assert old == [messages[1]]
    assert kept[0]["content"] == [{"type": "text", "text": "当前片段"}, video]
