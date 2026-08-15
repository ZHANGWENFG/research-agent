"""
STORM 语言模型抽象层
====================

这是整个 STORM 系统中**最核心的基础设施模块之一**，负责将 LLM 调用抽象为统一接口。
理解这个文件是理解"Agent 系统为什么要抽象 LM 层"的关键。

整体架构:
┌──────────────────────────────────────────────────────────────┐
│  STORMWikiRunner (engine.py)                                │
│  需要调用 LLM 做：对话模拟、大纲生成、文章撰写、润色等           │
│  但它不关心底层是哪个 model / provider                         │
└────────────────────┬─────────────────────────────────────────┘
                     │ 只依赖一个接口: lm(prompt) → [completions]
                     ▼
┌──────────────────────────────────────────────────────────────┐
│  LM 抽象层 (本文件)                                           │
│                                                              │
│  推荐方案: LitellmModel (v1.1.0+)                            │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ litellm 作为统一网关，支持 100+ providers:            │    │
│  │ openai/gpt-4    claude-3-opus    minimax/MiniMax-M3 │    │
│  │ 格式: "provider/model_name" 即可切换                   │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                              │
│  旧方案 (deprecated): 每个 provider 一个类                     │
│  ┌──────────┬───────────┬──────────┬──────────┬────────┐    │
│  │OpenAI    │DeepSeek   │Claude    │Gemini    │Groq... │    │
│  └──────────┴───────────┴──────────┴──────────┴────────┘    │
│  这些类保留只为向后兼容，新代码请用 LitellmModel                 │
└──────────────────────────────────────────────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────────────────────────┐
│  两层缓存系统 (节省 API 费用，加速调试)                          │
│  第1层: LRU 内存缓存 (LM_LRU_CACHE_MAX_SIZE=3000)              │
│  第2层: litellm 磁盘缓存 (~/.storm_local_cache)                │
│  相同 prompt → 直接返回缓存，不发 API 请求                       │
└──────────────────────────────────────────────────────────────┘

设计模式:
- 策略模式: 每个 Model 类封装一种 API 调用策略，对外接口一致
- 装饰器模式: @backoff.on_exception 实现自动重试
- 模板方法: LM.__call__ 定义调用流程，子类实现具体请求细节

学习要点:
1. 为什么要把 LM 抽象成独立模块？因为 Agent 系统中 LLM 是"可替换的零件"，
   不同的任务可能用不同的模型（快模型做对话模拟，强模型做文章生成）
2. litellm 在这里的作用 = 多 provider 的统一适配层，就像 USB Hub 一样
3. 两层缓存在开发调试时极其重要，否则每次跑 pipeline 都是巨额 API 费用
"""

import functools
import logging
import os
import random
import requests
import threading
from typing import Optional, Literal, Any
import ujson
from pathlib import Path


try:  # transformers 可选（HF 旧类使用；主线走 litellm 不需要）
    from transformers import AutoTokenizer
except Exception:  # noqa: BLE001
    AutoTokenizer = None

from openai import OpenAI, AzureOpenAI  # noqa: E402

# Anthropic 的 SDK 可能未安装，用 try/except 做可选依赖
try:
    from anthropic import RateLimitError
except ImportError:
    RateLimitError = None

############################
# 以下代码为 litellm 兼容层（缓存 + completion 基础函数）
# 用于提供基于 litellm 的缓存 + completion 基础函数
############################

import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=UserWarning)
    # LITELLM_LOCAL_MODEL_COST_MAP=True 让 litellm 在本地计算 token cost，
    # 而不是每次调 litellm 的远程 API 去查价格表（减少网络请求）
    if "LITELLM_LOCAL_MODEL_COST_MAP" not in os.environ:
        os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    import litellm

    # drop_params=True: 如果传给 litellm 的参数某个 provider 不支持，
    # 自动丢弃而不是报错（提高兼容性）
    litellm.drop_params = True
    # 关闭 litellm 的遥测数据上报
    litellm.telemetry = False

from litellm.caching.caching import Cache

# 磁盘缓存目录: ~/.storm_local_cache
# 注意: 这是 litellm 层面的缓存，与下面的 LRU 内存缓存是两个独立层
disk_cache_dir = os.path.join(Path.home(), ".storm_local_cache")
litellm.cache = Cache(disk_cache_dir=disk_cache_dir, type="disk")

# 注释掉的代码是 litellm 未安装时的 fallback 处理
# 因为 litellm 已在 requirements.txt 中，所以直接 import
# except ImportError:
#     class LitellmPlaceholder:
#         def __getattr__(self, _):
#             raise ImportError(...)
# litellm = LitellmPlaceholder()

# LRU 内存缓存最大条目数: 3000
# 注意: LRU cache 的 key 是 request JSON 字符串，所以相同 prompt+参数会命中缓存
LM_LRU_CACHE_MAX_SIZE = 3000


# ====================================================================
# LM — 基础抽象类
# ====================================================================
# 这是所有语言模型类的基类。它定义了 LLM 调用的统一接口:
#   - __call__(prompt, messages, **kwargs) → [completions]
#   - history: 记录每次调用的 prompt/response/usage/cost
#   - cache: 控制是否使用缓存
#
# 关键设计: __call__ 中用 ujson.dumps 把请求参数序列化为字符串，
# 然后传给 cached_litellm_completion。这样做的好处是:
#   - LRU cache 可以直接用字符串当 key
#   - 确保完全相同的请求参数才能命中缓存
# ====================================================================

class LM:
    def __init__(
        self,
        model,
        model_type="chat",  # "chat" 或 "text"，走 litellm 的不同 API 端点
        temperature=0.0,     # STORM 场景不需要创造性，默认 0.0
        max_tokens=1000,
        cache=True,          # 默认开启缓存（省钱 + 可复现）
        **kwargs,
    ):
        self.model = model
        self.model_type = model_type
        self.cache = cache
        self.kwargs = dict(temperature=temperature, max_tokens=max_tokens, **kwargs)
        self.history = []  # 调用历史，调试和 token 统计用

        # OpenAI o1 系列模型有特殊要求: 温度必须为 1.0，max_tokens >= 5000
        if "o1-" in model:
            assert (
                max_tokens >= 5000 and temperature == 1.0
            ), "OpenAI's o1-* models require passing temperature=1.0 and max_tokens >= 5000 to this LM"

    def __call__(self, prompt=None, messages=None, **kwargs):
        """
        LLM 调用的统一入口。

        调用流程:
        1. 组装 messages（如果传了 prompt 则包装为 user message）
        2. 根据 model_type 选择 chat/text completion 函数
        3. 根据 cache 标志选择是否走缓存
        4. 调用 litellm → 解析响应 → 记录 history → 返回 output 列表
        """
        # 第1行: dict.pop(key, default) — 从 kwargs 中取出 "cache" 并删除，
        # 如果调用者没传 cache 参数，则回退到实例默认值 self.cache
        cache = kwargs.pop("cache", self.cache)

        # 第2行: Python 的短路 or — 如果 messages 为 None/空列表等 falsy 值，
        # 就用 prompt 构造一个标准的 user message
        # 等价于: messages = messages if messages else [{"role": "user", "content": prompt}]
        messages = messages or [{"role": "user", "content": prompt}]

        # 第3行: ** 字典解包合并 — 把 self.kwargs（实例默认参数）和 kwargs（调用时传入的参数）
        # 合并成一个新字典，kwargs 的键会覆盖 self.kwargs 中的同名键
        # 例如 self.kwargs = {"temperature": 0.0, "max_tokens": 500}
        #      kwargs     = {"temperature": 1.0}  （调用时传入）
        # 合并后 → {"temperature": 1.0, "max_tokens": 500}
        kwargs = {**self.kwargs, **kwargs}

        # 选择缓存的或直接的 completion 函数，#一般用chat和cache，缓存命中省钱
        if self.model_type == "chat":
            completion = cached_litellm_completion if cache else litellm_completion
        else:
            completion = (
                cached_litellm_text_completion if cache else litellm_text_completion
            )

        # === 调用 LLM ===
        # ujson.dumps(dict(...)) 把请求参数序列化为 JSON 字符串，
        # 这个字符串同时充当 LRU 缓存的 key（相同参数 → 相同字符串 → 命中缓存）
        response = completion(
            ujson.dumps(dict(model=self.model, messages=messages, **kwargs))
        )

        # === 解析响应，提取文本 ===
        # response["choices"] 是一个列表，每个元素是一个候选回复
        # 列表推导式: 遍历每个 choice，提取里面的文本
        #
        # hasattr(c, "message") 区分两种响应格式:
        #   chat 模式   → c 是对象，文本在 c.message.content
        #   text 模式   → c 是字典，文本在 c["text"]
        # 这行代码保证无论用哪种 model_type，都能正确拿到文本
        outputs = [
            c.message.content if hasattr(c, "message") else c["text"]
            for c in response["choices"]
        ]

        # 记录调用历史（去掉 api_key 等敏感信息后再记录）
        kwargs = {k: v for k, v in kwargs.items() if not k.startswith("api_")}
        entry = dict(prompt=prompt, messages=messages, kwargs=kwargs, response=response)
        entry = dict(**entry, outputs=outputs, usage=dict(response["usage"]))
        entry = dict(
            **entry, cost=response.get("_hidden_params", {}).get("response_cost")
        )
        self.history.append(entry)

        return outputs

    def inspect_history(self, n: int = 1):
        """打印最近 n 次 LLM 调用的 prompt 和 completion，调试用"""
        _inspect_history(self, n)


# ====================================================================
# 缓存层实现
# ====================================================================
# 两层缓存设计:
#
# 第1层: @functools.lru_cache (内存)
#   - 快速，进程内共享
#   - key = request JSON 字符串
#   - 最多 3000 条 (LM_LRU_CACHE_MAX_SIZE)
#   - 注意: litellm 的 response 对象需要支持 hash，所以用 ujson.dumps 序列化
#
# 第2层: litellm.cache (磁盘)
#   - 跨进程持久化，重启后仍有效
#   - 存储目录: ~/.storm_local_cache
#   - litellm 原生支持，无需额外代码
#
# 当 cache=False 时，两级缓存都不走（直接调 API）
# ====================================================================

@functools.lru_cache(maxsize=LM_LRU_CACHE_MAX_SIZE)
def cached_litellm_completion(request):
    """
    第1层缓存: LRU 内存缓存
    request 是 ujson.dumps 序列化的参数字符串，相同参数一定会命中缓存
    缓存命中后不再调用 litellm，直接返回之前的 response 对象
    """
    return litellm_completion(request, cache={"no-cache": False, "no-store": False})


def litellm_completion(request, cache={"no-cache": True, "no-store": True}):
    """
    直接调用 litellm chat completion（不走 LRU 缓存时使用）
    cache={"no-cache": True} 的含义:
      - 不读 litellm 磁盘缓存
      - 但会把结果写入磁盘缓存（下次 LRU miss 时可以读到）
    """
    kwargs = ujson.loads(request)
    return litellm.completion(cache=cache, **kwargs)


@functools.lru_cache(maxsize=LM_LRU_CACHE_MAX_SIZE)
def cached_litellm_text_completion(request):
    """LRU 缓存的 text completion 版本"""
    return litellm_text_completion(
        request, cache={"no-cache": False, "no-store": False}
    )


def litellm_text_completion(request, cache={"no-cache": True, "no-store": True}):
    """
    直接调用 litellm text completion（用于 model_type="text" 的场景）
    与 chat completion 的区别:
      - text completion 是老式 API，直接给 prompt 字符串
      - chat completion 是对话式 API，给 messages 列表
    """
    kwargs = ujson.loads(request)

    # 从 model 字符串解析 provider 和 model 名
    # 例如 "openai/gpt-4o" → provider="openai", model="gpt-4o"
    # 如果没有 "/" 则默认 provider 为 "openai"
    model = kwargs.pop("model").split("/", 1)
    provider, model = model[0] if len(model) > 1 else "openai", model[-1]

    # API key 优先级: kwargs 显式传入 > 环境变量 {PROVIDER}_API_KEY
    api_key = kwargs.pop("api_key", None) or os.getenv(f"{provider}_API_KEY")
    api_base = kwargs.pop("api_base", None) or os.getenv(f"{provider}_API_BASE")

    # text completion 需要把 messages 拼成纯文本 prompt
    prompt = "\n\n".join(
        [x["content"] for x in kwargs.pop("messages")] + ["BEGIN RESPONSE:"]
    )

    return litellm.text_completion(
        cache=cache,
        model=f"text-completion-openai/{model}",
        api_key=api_key,
        api_base=api_base,
        prompt=prompt,
        **kwargs,
    )


# ====================================================================
# 调试工具函数
# ====================================================================

def _green(text: str, end: str = "\n"):
    """终端绿色输出（用于显示 LLM 回复）"""
    return "\x1b[32m" + str(text).lstrip() + "\x1b[0m" + end


def _red(text: str, end: str = "\n"):
    """终端红色输出（用于显示 prompt）"""
    return "\x1b[31m" + str(text) + "\x1b[0m" + end


def _inspect_history(lm, n: int = 1):
    """打印最近 n 次 LLM 调用的完整 prompt 和 completion，用于调试"""

    for item in lm.history[-n:]:
        messages = item["messages"] or [{"role": "user", "content": item["prompt"]}]
        outputs = item["outputs"]

        print("\n\n\n")
        for msg in messages:
            print(_red(f"{msg['role'].capitalize()} message:"))
            print(msg["content"].strip())
            print("\n")

        print(_red("Response:"))
        print(_green(outputs[0].strip()))

        if len(outputs) > 1:
            choices_text = f" \t (and {len(outputs)-1} other completions)"
            print(_red(choices_text, end=""))

    print("\n\n\n")


############################


# ====================================================================
# LitellmModel — 推荐的 LLM 封装类 (v1.1.0+)
# ====================================================================
# 这是 STORM 当前推荐的 LLM 封装类，底层直接走 litellm。
#
# 为什么推荐用它而不是下面的旧类？
# 1. litellm 统一了 100+ provider 的接口差异，新增 provider 无需写新类
# 2. 支持所有 litellm 特性: 自动 fallback、cost tracking、streaming 等
# 3. 模型切换只需改 model 字符串: "minimax/MiniMax-M3" → 切到 MiniMax
#
# 使用方式:
#   lm = LitellmModel(model="openai/gpt-4o", api_key="...", temperature=0.0)
#   outputs = lm("What is AI?")  # → ["AI is ..."]
# ====================================================================

class LitellmModel(LM):
    """基于 litellm 的统一 LLM 封装，支持所有 litellm 兼容的 provider。

    用法: LitellmModel(model="provider/model_name", api_key="...")
    参考: https://docs.litellm.ai/docs/providers
    """

    def __init__(
        self,
        model: str = "openai/gpt-4o-mini",
        api_key: Optional[str] = None,
        model_type: Literal["chat", "text"] = "chat",
        **kwargs,
    ):
        super().__init__(model=model, api_key=api_key, model_type=model_type, **kwargs)
        # token 统计: 线程安全的计数器（因为 STORM 可能多线程并发调 LLM）
        self._token_usage_lock = threading.Lock()
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def log_usage(self, response):
        """从 litellm 响应中提取 token 用量并累加到计数器"""
        usage_data = response.get("usage")
        if usage_data:
            with self._token_usage_lock:
                self.prompt_tokens += usage_data.get("prompt_tokens", 0)
                self.completion_tokens += usage_data.get("completion_tokens", 0)

    def get_usage_and_reset(self):
        """
        获取累计 token 用量并重置计数器。
        这个方法在 STORM pipeline 每个阶段结束时被调用，用于统计各阶段消耗。
        返回格式: {"model_name": {"prompt_tokens": N, "completion_tokens": M}}
        """
        usage = {
            self.model
            or self.kwargs.get("model")
            or self.kwargs.get("engine"): {
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
            }
        }
        self.prompt_tokens = 0
        self.completion_tokens = 0

        return usage

    def __call__(self, prompt=None, messages=None, **kwargs):
        """
        调用 LLM 并返回 completion 列表。
        与父类 LM.__call__ 的区别:
        - 额外调用 log_usage() 统计 token
        - 解析 response.json() 而不是直接使用 response 对象
        """
        # dict.pop(key, default) — 取出并删除 "cache" 键，未传入则用实例默认值
        cache = kwargs.pop("cache", self.cache)

        # 短路 or — messages 为空时用 prompt 构造 user message
        messages = messages or [{"role": "user", "content": prompt}]

        # ** 字典合并 — 实例默认参数(左) + 调用参数(右)，调用参数覆盖同名键
        kwargs = {**self.kwargs, **kwargs}

        if self.model_type == "chat":
            completion = cached_litellm_completion if cache else litellm_completion
        else:
            completion = (
                cached_litellm_text_completion if cache else litellm_text_completion
            )

        response = completion(
            ujson.dumps(dict(model=self.model, messages=messages, **kwargs))
        )
        response_dict = response.json()
        self.log_usage(response_dict)
        outputs = [
            c.message.content if hasattr(c, "message") else c["text"]
            for c in response["choices"]
        ]

        kwargs = {k: v for k, v in kwargs.items() if not k.startswith("api_")}
        entry = dict(
            prompt=prompt, messages=messages, kwargs=kwargs, response=response_dict
        )
        entry = dict(**entry, outputs=outputs, usage=dict(response_dict["usage"]))
        entry = dict(
            **entry, cost=response.get("_hidden_params", {}).get("response_cost")
        )
        self.history.append(entry)

        return outputs


# ========================================================================
# 以下所有模型类在 v1.1.0 后均已废弃 (deprecated)
# ========================================================================
