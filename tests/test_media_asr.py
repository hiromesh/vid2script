import math

from vid2script.asr import parse_transcript
from vid2script.media import estimate_video_tokens


def test_video_token_estimate_uses_frame_area():
    expected = math.ceil(2 * 3 * 320 * 240 / 3136 * 1.15)
    assert estimate_video_tokens(2, 320, 240, 3) == expected


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
    assert all(cue.end - cue.start <= 6 for cue in cues)
    assert "".join(cue.text for cue in cues) == "".join(word["text"] for word in words)
