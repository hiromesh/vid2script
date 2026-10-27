import asyncio
import json
import logging
import os
from copy import deepcopy
from pathlib import Path

import tiktoken
from openai import AsyncOpenAI

from .config import Settings
from .script import dump

log = logging.getLogger(__name__)
PROMPTS = Path(__file__).parent / "prompts"


class InputTooLarge(ValueError):
    pass


class OutputTruncated(RuntimeError):
    pass


def text_only(messages: list[dict], keep_reasoning: bool = True) -> list[dict]:
    output = deepcopy(messages)
    for message in output:
        content = message.get("content")
        if isinstance(content, list):
            message["content"] = [part for part in content if part.get("type") == "text"]
        if not keep_reasoning:
            message.pop("reasoning_content", None)
    return output


def compact_parts(messages: list[dict], turn_start: int) -> tuple[list[dict], list[dict]]:
    old = messages[1:turn_start]
    current = messages[turn_start:]
    exchanges = [i for i, m in enumerate(current) if m["role"] == "assistant"]
    if exchanges:
        # 保留当前任务和最后一次完整调用；此前已经完成的当前段交互也可以整理。
        split = exchanges[-1]
        old = [*old, *text_only(current[:split], keep_reasoning=False)]
        kept = text_only([current[0], *current[split:]])
    else:
        kept = current
    return text_only(old, keep_reasoning=False), kept


class Model:
    def __init__(self, settings: Settings, work: Path):
        self.settings = settings
        cache = work / "tokenizer"
        cache.mkdir(parents=True, exist_ok=True)
        os.environ["TIKTOKEN_CACHE_DIR"] = str(cache.resolve())
        self.encoding = tiktoken.get_encoding("o200k_base")
        self.client = AsyncOpenAI(api_key=settings.api_key.get_secret_value(), base_url=settings.base_url,
                                  timeout=settings.request_timeout_seconds, max_retries=2)
        self.scale = 1.0
        self.calls = self.compactions = self.prompt_tokens = self.completion_tokens = 0
        self.peak_estimated_tokens = self.peak_request_bytes = self.peak_videos = 0

    def estimate(self, messages: list[dict], tools: list[dict], extra_tokens: int = 0) -> int:
        text = dump({"messages": text_only(messages), "tools": tools})
        return int((len(self.encoding.encode(text, disallowed_special=())) + extra_tokens +
                    12 * len(messages)) * self.scale)

    async def call(self, messages: list[dict], tools: list[dict] | None = None,
                   output_tokens: int | None = None, compact: bool = False, extra_tokens: int = 0):
        budget = output_tokens or self.settings.output_budget
        tools = tools or []
        predicted = self.estimate(messages, tools, extra_tokens)
        videos = sum(1 for m in messages if isinstance(m.get("content"), list)
                     for b in m["content"] if b.get("type") == "video_url")
        payload = {"model": self.settings.model, "messages": messages, "max_tokens": budget,
                   "stream": False}
        if tools:
            payload.update(tools=tools, tool_choice="auto", parallel_tool_calls=False)
        thinking = self.settings.compact_thinking_mode if compact else self.settings.thinking_mode
        if thinking != "default":
            payload["extra_body"] = {"thinking": {"type": thinking}}
        if self.settings.reasoning_effort and thinking != "disabled":
            payload["reasoning_effort"] = self.settings.reasoning_effort
        request_bytes = len(json.dumps(payload, ensure_ascii=True).encode())
        if predicted + budget + 1024 > self.settings.context_window_tokens:
            raise InputTooLarge("当前输入与输出预留超过上下文容量")
        if videos > self.settings.max_videos or request_bytes > self.settings.max_request_bytes:
            raise InputTooLarge("当前请求超过视频数量或请求体预算")
        async with asyncio.timeout(self.settings.request_timeout_seconds):
            response = await self.client.chat.completions.create(**payload)
        self.calls += 1
        self.peak_estimated_tokens = max(self.peak_estimated_tokens, predicted)
        self.peak_request_bytes = max(self.peak_request_bytes, request_bytes)
        self.peak_videos = max(self.peak_videos, videos)
        if response.usage:
            actual = response.usage.prompt_tokens
            self.prompt_tokens += actual
            self.completion_tokens += response.usage.completion_tokens
            self.scale = max(self.scale, min(4.0, self.scale * actual / max(1, predicted) * 1.05))
        choice = response.choices[0]
        if choice.finish_reason == "length":
            raise OutputTruncated("模型输出达到上限，本次结果未提交")
        if choice.finish_reason not in {"stop", "tool_calls"}:
            raise RuntimeError(f"模型未正常完成：{choice.finish_reason}")
        return choice.message

    async def close(self) -> None:
        await self.client.close()


class History:
    def __init__(self, model: Model):
        self.model = model
        prompt = (PROMPTS / "agent.md").read_text()
        self.messages = [{"role": "system", "content": prompt}]
        self.extra_tokens = 0
        self.turn_start = 1

    def begin(self, content: list[dict], extra_tokens: int = 0) -> None:
        self.messages = text_only(self.messages)
        self.extra_tokens = extra_tokens
        self.turn_start = len(self.messages)
        self.messages.append({"role": "user", "content": content})

    async def prepare(self, tools: list[dict], state: dict) -> None:
        size = self.model.estimate(self.messages, tools, self.extra_tokens)
        capacity = self.model.settings.context_window_tokens
        if size < capacity * 0.7 and size + self.model.settings.output_budget + 1024 <= capacity:
            return
        old, kept = compact_parts(self.messages, self.turn_start)
        if not old:
            raise InputTooLarge("当前一段输入过大，需要缩小范围")
        summary_input = [{"role": "system", "content": (PROMPTS / "compact.md").read_text()},
                         {"role": "user", "content": dump({"history": old, "current_state": state})}]
        result = await self.model.call(summary_input, output_tokens=3072, compact=True)
        summary = result.content
        if not summary or len(self.model.encoding.encode(summary, disallowed_special=())) > 3072:
            raise RuntimeError("compact 摘要为空或超过预算，原历史仍然保留")
        self.messages = [self.messages[0], {"role": "user", "content": "<compact-summary>\n" + summary +
                                          "\n</compact-summary>"}, *kept]
        self.turn_start = 2
        self.extra_tokens = 0
        self.model.compactions += 1
        log.info("compact %s → %s estimated tokens", size, self.model.estimate(self.messages, tools))
        if self.model.estimate(self.messages, tools) >= capacity * 0.7:
            raise InputTooLarge("compact 后当前片段仍超过上下文预算")

    def assistant(self, response) -> None:
        message = {"role": "assistant", "content": response.content}
        if response.tool_calls:
            message["tool_calls"] = [call.model_dump(exclude_none=True) for call in response.tool_calls]
        reasoning = getattr(response, "reasoning_content", None)
        if reasoning:
            message["reasoning_content"] = reasoning
        self.messages.append(message)

    def result(self, call_id: str, content: dict) -> None:
        self.messages.append({"role": "tool", "tool_call_id": call_id, "content": dump(content)})
