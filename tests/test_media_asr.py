from itertools import pairwise
from pathlib import Path

from vid2script.asr import parse_transcript
from vid2script.llm import text_only
from vid2script.media import Video, timestamps


def test_three_hour_sampling_is_global_and_does_not_densify_at_batch_edges():
    video = Video(Path("movie.mp4"), 10800, 1920, 1080, True, None)
    whole = timestamps(video, 0, 10800)
    parts = timestamps(video, 0, 31) + timestamps(video, 31, 61) + timestamps(video, 61, 10800)
    assert len(whole) == 5400 and whole == parts
    assert all(b - a == 2 for a, b in pairwise(whole))


def test_short_video_uses_one_second_grid():
    video = Video(Path("short.mp4"), 19.5, 1080, 1920, False, None)
    assert timestamps(video, 0, 19.5) == list(range(20))


def test_asr_scopes_speakers_and_offsets_timestamps():
    body = {"result": {"utterances": [{"start_time": 100, "end_time": 900, "text": "你好。",
                                      "additions": '{"speaker":"1"}'}]}}
    first = parse_transcript(body, 0, 10, 10, 12)[0]
    second = parse_transcript(body, 1, 20, 20, 22)[0]
    assert first.start == 10.1 and first.end == 10.9
    assert first.speaker != second.speaker and first.id != second.id


def test_asr_overlap_assigns_each_word_once():
    body = {"result": {"utterances": [{"start_time": 0, "end_time": 4000, "text": "前半后半",
        "words": [{"start_time": 0, "end_time": 1900, "text": "前半"},
                  {"start_time": 2100, "end_time": 4000, "text": "后半"}]}]}}
    first = parse_transcript(body, 0, 0, 0, 2)
    second = parse_transcript(body, 1, 0, 2, 4)
    assert "".join(c.text for c in first + second) == "前半后半"


def test_long_asr_utterance_splits_into_bounded_cues():
    words = [{"start_time": i * 1000, "end_time": i * 1000 + 900, "text": str(i)} for i in range(90)]
    cues = parse_transcript({"result": {"utterances": [
        {"start_time": 0, "end_time": 89900, "text": "长对白", "words": words}
    ]}}, 0, 0, 0, 90)
    assert len(cues) == 15
    assert all(c.end - c.start <= 6 for c in cues)
    assert "".join(c.text for c in cues) == "".join(w["text"] for w in words)


def test_image_eviction_does_not_remove_tool_pairs_or_reasoning():
    messages = [
        {"role": "user", "content": [{"type": "text", "text": "f1"},
                                       {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,AAAA"}}]},
        {"role": "assistant", "reasoning_content": "推理", "tool_calls": [{"id": "call1"}]},
        {"role": "tool", "tool_call_id": "call1", "content": "已提交"},
    ]
    kept = text_only(messages)
    assert kept[0]["content"] == [{"type": "text", "text": "f1"}]
    assert kept[1:] == messages[1:]
    assert "reasoning_content" not in text_only(messages, keep_reasoning=False)[1]
    assert len(messages[0]["content"]) == 2
