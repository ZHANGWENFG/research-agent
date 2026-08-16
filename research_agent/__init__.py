# my-agent research_agent 包入口（显式导出主线类）
# 2026-08-16: 移除星号导入（F403）——历史星号导入使 lint 无法检测 undefined name，
# 且会把无关符号拖进命名空间；无任何代码依赖 `from research_agent import *`。
from .lm import LM, LitellmModel
from .rm import ArxivRM, LocalPDFRM

__all__ = ["LM", "LitellmModel", "ArxivRM", "LocalPDFRM"]
__version__ = "1.0.0"
