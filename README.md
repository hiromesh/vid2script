# vid2script

将本地 MP4、MOV 等视频转换为演员可读的 Markdown 剧本，支持单集、连续多集和电影。

使用一个 agent 串行观察、补充核对、增量写作与 compact。剧本按场景排列，动作与对白交错；角色身份未确认时使用临时称呼，不凭剧情补写缺失台词。

观察材料是原视频的无声片段：每批最长 30 秒，在本地用 FFmpeg 压缩成 MP4，再以 base64 `video_url` 发给模型。模型直接看连续画面，正文用 `video_ids` 引用片段证据。

## 运行

需要 Python 3.12+、FFmpeg 和 FFprobe。

```bash
.venv/bin/pip install -e '.[dev]'
.venv/bin/vid2script run /path/01.mp4 --title 剧名
.venv/bin/vid2script run /path/01.mp4 /path/02.mp4 --title 剧名
.venv/bin/vid2script run /path/movie.mp4 --title 电影名 --movie
```

多集按照命令参数的顺序处理。`--first-episode` 指定起始集数。输出为 `output/剧名/第01集.md`，电影输出为 `output/电影名/电影.md`。已有文件默认拒绝覆盖，确认替换时加 `--overwrite`。

每批片段默认编码为长边 640、3fps、CRF 30 的无声 MP4，直接嵌入请求，不上传 OSS；可通过 `VIDEO_LONG_EDGE`、`VIDEO_FPS`、`VIDEO_CRF` 调整。片段起止时间写入 `clip_id`，正文用 `video_ids` 引用它。对白仍然只来自字幕或 ASR；片段移除了音轨，画面中的硬字幕可作为人名与台词的核对证据。

只测试原视频片段：

```bash
.venv/bin/vid2script --env .work/test.env run /path/01.mp4 \
  --title 片段测试 --start 180 --end 240
```

时间参数采用原视频的秒数，成稿保留原片时间。片段边界可能位于一句对白中间，处理范围会注明在 Markdown 中。

外部字幕使用 `--subtitle /path/subtitle.srt`，也自动识别同名 `.srt` 和内嵌文本字幕。其他 FFmpeg 支持的文本字幕格式可以通过 `--subtitle` 指定。没有文本字幕且存在音轨时使用字节跳动 ASR；没有音轨时依据画面及可见字幕生成，并由 agent 标注信息不足处。

## 配置

项目根目录 `.env` 提供主模型，示例见 `.env.example`。进程环境变量优先于 `.env`。主模型需支持 Chat Completions、`video_url` 中的 base64 data URL 和工具调用；示例配置使用 `qwen3.8-flash`，可用 `--model` 临时覆盖。

| 配置 | 作用 |
| --- | --- |
| `MODEL`、`BASE_URL`、`API_KEY` | 主模型与网关 |
| `VIDEO_LONG_EDGE`、`VIDEO_FPS`、`VIDEO_CRF` | 视频片段编码参数，默认 640、3fps、CRF 30 |
| `VIDEO_MAX_BYTES` | 单个视频片段压缩后的字节上限，默认 8 MiB |
| `MAX_VIDEO_SECONDS`、`MAX_VIDEOS` | 单个片段最长秒数与单次请求视频数量，默认 30 秒、3 段 |
| `MAX_REQUEST_BYTES` | 单次请求序列化字节预算，默认 24 MiB |
| `REQUEST_TIMEOUT_SECONDS` | 单次模型请求超时，默认 240 秒 |
| `MAX_OUTPUT_TOKENS` | 服务端输出允许上限；实际每次写作最多使用 8192 tokens，并按上下文预算进一步限制 |
| `THINKING_MODE` | `default` 使用模型默认值；支持该参数的模型可配置 `enabled` 或 `disabled` |
| `REASONING_EFFORT` | 默认不发送；支持的模型可设置 `low`、`high`、`max` |
| `COMPACT_THINKING_MODE` | 同一模型整理摘要时默认 `disabled`，将输出预算留给摘要；不支持该参数的服务配置为 `default` |
| `CONTEXT_WINDOW_TOKENS` | 本工具使用的上下文预算，默认 131072；不要超过网关实际支持值 |
| `BATCH_SECONDS` | 默认每批 30 秒；输入预算不足时自动缩短 |
| `MAX_TURNS_PER_BATCH` | 每批最多工具调用轮数，默认 30 |
| `ASR_SEGMENT_SECONDS` | 音频分段目标长度，默认 300 秒，在附近静音处切分并保留边界重叠 |
| `ASR_POLL_INTERVAL_SECONDS` | ASR 任务轮询间隔，默认 2 秒 |
| `VIDEO_ANALYSIS__ASR_API_KEY` | 字节 ASR 凭据 |
| `VIDEO_ANALYSIS__ASR_RESOURCE_ID` | 默认 `volc.bigasr.auc` |
| `VIDEO_ANALYSIS__ASR_TIMEOUT_SECONDS` | 识别任务等待时间，默认 900 秒 |
| `OSS__ENDPOINT`、`OSS__BUCKET` | 阿里云 OSS 配置 |
| `OSS__ACCESS_KEY_ID`、`OSS__ACCESS_KEY_SECRET` | 上传、签名访问和删除临时音频的凭据 |

从已有项目导入服务配置，只复制 OSS 与 ASR 字段，不覆盖当前非空值：

```bash
.venv/bin/vid2script import-config /path/reference/.env
```

视频片段经本地压缩后随模型请求发送，不上传 OSS。超过单片段字节上限时自动降低一档分辨率、帧率和压缩质量重试。音频使用 OSS 临时对象和有期限的签名 URL，识别完成或报错后执行删除。ASR 失败会中止，不能将失败当作无对白。

## 长视频处理

- 按批生成无声视频片段，默认最长 30 秒；片段起止时间写入 `clip_id`，正文用 `video_ids` 引用画面证据。逐页核对按页面范围取最多 3 段短视频。
- 每次只读取当前片段；MP4 编码完成后才转为 base64。已完成片段的视频从消息中移除，需要回看时重新提取。同一片段在多轮工具调用中会命中网关的前缀缓存。
- `update_script` 每次最多提交 12 条正文，只修改一个场景。长场景允许无限次追加。操作 ID 保证同一次提交重试不会重复写入；参数验证通过后才原子更新正文。
- 观察范围和写入完成分别管理。台词必须引用或明确说明排除原因，全部处理后才能推进批次。
- 输入预计达到上下文的 70% 时，用同一个模型串行生成 compact 摘要。保留完整的当前交互，工具调用与结果不会被拆开。摘要中不包含 base64。
- 完整结构化剧本、人物和台词保存在内存中，compact 只修改 `messages`。长场景的当前上下文仅带最近正文和连续性摘要，历史正文可以分页读取。
- 核对按正文页执行，最终由 Python 渲染完整 Markdown；不让模型重新输出整部影片。
- 每批完成后原子更新 Markdown，结束前标注“处理中”。异常或中止保留已完成正文，不声称任务已完成。会话及运行状态不持久化，不提供断点续跑。
- 跨集保留人物资料和必要连续性信息，旧集正文通过已生成的 Markdown 分页读取。

本地临时媒体在 `.work/` 下，任务结束后清理；tokenizer 缓存保留在 `.work/tokenizer/`。`.env`、`.work/`、`output/` 都已加入 `.gitignore`。

## 测试

```bash
.venv/bin/pytest -q
.venv/bin/ruff check .
```

本地测试验证长场景多次增量写入、台词遗漏检查、重复提交、局部修订、ASR 时间偏移和重叠归属、视频片段编码与裁剪、`video_ids` 证据、上下文整理边界和分批边界。它们不使用模型或服务 mock。

真实服务检查会调用模型、OSS 与字节 ASR，并产生相应 API 用量：

```bash
.venv/bin/python scripts/check_services.py --env .work/test.env --video /path/01.mp4
```

服务检查需要带音轨的视频，默认读取前 10 秒，可设置 `--start 180 --seconds 30`；端到端片段测试使用上面的 `--start` 和 `--end` 命令。
