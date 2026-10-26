import argparse
import asyncio
import json
import logging
import re
from pathlib import Path
from tempfile import TemporaryDirectory

from .agent import Agent
from .asr import Speech, Transcript
from .config import Settings, import_services
from .llm import Model
from .media import Media, probe
from .script import Script


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="使用单一 agent 将视频转换为演员剧本 Markdown")
    root.add_argument("--env", type=Path, default=Path(".env"))
    commands = root.add_subparsers(dest="command", required=True)
    config = commands.add_parser("import-config", help="从参考 .env 导入 OSS/ASR 配置，不覆盖现有非空值")
    config.add_argument("source", type=Path)
    process = commands.add_parser("run", help="按提供的顺序处理一集、多集或电影")
    process.add_argument("videos", nargs="+", type=Path)
    process.add_argument("--title", default="")
    process.add_argument("--movie", action="store_true", help="单个视频作为电影输出")
    process.add_argument("--first-episode", type=int, default=1)
    process.add_argument("--output", type=Path, default=Path("output"))
    process.add_argument("--work", type=Path, default=Path(".work"))
    process.add_argument("--subtitle", type=Path, help="单个视频的外部字幕；未指定时检查同名 .srt 和内嵌字幕")
    process.add_argument("--start", type=float, default=0, help="原视频秒数，默认从头处理")
    process.add_argument("--end", type=float, help="原视频结束秒数，默认处理完整视频")
    process.add_argument("--overwrite", action="store_true")
    return root


async def process(args: argparse.Namespace) -> None:
    if len(args.videos) > 1 and (args.movie or args.subtitle or args.start or args.end is not None):
        raise ValueError("movie、subtitle 和时间范围参数仅适用于单个视频")
    if args.first_episode < 1:
        raise ValueError("集数必须为正数")
    settings = Settings.load(args.env)
    args.work.mkdir(parents=True, exist_ok=True)
    model = Model(settings, args.work)
    agent = Agent(model)
    title = args.title or (args.videos[0].stem if args.movie else args.videos[0].parent.name)
    folder = args.output / safe_name(title)
    try:
        for index, path in enumerate(args.videos, args.first_episode):
            episode = "电影" if args.movie else f"第{index:02}集"
            target = folder / f"{episode}.md"
            if target.exists() and not args.overwrite:
                raise FileExistsError(f"输出已存在：{target}；需要替换时传入 --overwrite")
            video = await probe(path.resolve())
            stop = min(video.duration, args.end) if args.end is not None else video.duration
            if not 0 <= args.start < stop:
                raise ValueError("处理时间范围无效")
            with TemporaryDirectory(dir=args.work, prefix="episode-") as directory:
                media = Media(video, Path(directory), settings)
                sidecar = path.with_suffix(".srt")
                subtitles = await media.subtitles(args.subtitle or (sidecar if sidecar.exists() else None))
                speech = Speech(settings) if subtitles is None and video.audio else None
                transcript = Transcript(media, speech, subtitles, stop)
                if args.start > 0 and subtitles is None and video.audio:
                    transcript.ready_until = args.start
                script = Script(title, episode)
                await agent.episode(media, transcript, script, target, (args.start, stop))
            print(f"已保存：{target.resolve()}", flush=True)
    finally:
        await model.close()
        stats = {key: getattr(model, key) for key in (
            "calls", "compactions", "prompt_tokens", "completion_tokens", "peak_estimated_tokens",
            "peak_request_bytes", "peak_images")}
        logging.getLogger(__name__).info("运行统计 %s", json.dumps(stats))


def safe_name(value: str) -> str:
    name = re.sub(r'[\\/:*?"<>|\x00-\x1f]', "_", value).strip(" .")
    if not name:
        raise ValueError("剧名不能为空")
    return name


def main() -> None:
    args = parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpx2").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("oss2").setLevel(logging.WARNING)
    if args.command == "import-config":
        print("已导入字段：", ", ".join(import_services(args.source, args.env)))
        return
    try:
        asyncio.run(process(args))
    except KeyboardInterrupt:
        raise SystemExit("任务已中止；已写出的 Markdown 保留为未完成状态") from None


if __name__ == "__main__":
    main()
