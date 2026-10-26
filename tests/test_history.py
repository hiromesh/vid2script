from vid2script.llm import compact_parts


def test_compact_boundary_preserves_last_tool_exchange_and_active_task():
    # 直接检查分界算法，不调用或替代任何模型服务。
    messages = [
        {"role": "system", "content": "规则"},
        {"role": "user", "content": "上一段"},
        {"role": "assistant", "tool_calls": [{"id": "old"}]},
        {"role": "tool", "tool_call_id": "old", "content": "已写入"},
        {"role": "user", "content": [{"type": "text", "text": "当前片段与台词"},
                                      {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,AAAA"}}]},
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
