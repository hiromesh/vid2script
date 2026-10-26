import argparse
import asyncio
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from vid2script.asr import Speech
from vid2script.config import Settings
from vid2script.llm import Model
from vid2script.media import Media, probe


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=Path, default=Path(".env"))
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--start", type=float, default=0)
    parser.add_argument("--seconds", type=float, default=10)
    args = parser.parse_args()
    settings = Settings.load(args.env)
    work = Path(".work")
    work.mkdir(exist_ok=True)
    video = await probe(args.video)
    model = Model(settings, work)
    try:
        with TemporaryDirectory(dir=work, prefix="service-check-") as directory:
            media = Media(video, Path(directory), settings)
            start, end = args.start, min(video.duration, args.start + args.seconds)
            path = await media.audio(start, end)
            speech = Speech(settings)
            cues = await speech.transcribe(path, 0, (start, start, end))
            print("ASR 台词数：", len(cues))
            print("ASR 示例：", [c.model_dump() for c in cues[:3]])
            print("待清理 OSS 对象：", len(speech.active_keys))
            images, _ = await media.images(start, min(end, start + video.interval))
            response = await model.call([{"role": "user", "content": [
                {"type": "text", "text": "请描述这张图片中可见的人物、场景和字幕，无法看清的内容不要猜测。"},
                *images,
            ]}], output_tokens=1024)
            print("图片输入结果：", response.content)
            (work / "service_check.json").write_text(json.dumps({
                "vision": response.content, "cues": [c.model_dump() for c in cues],
                "oss_remaining": len(speech.active_keys),
            }, ensure_ascii=False, indent=2))
    finally:
        await model.close()


if __name__ == "__main__":
    asyncio.run(main())
