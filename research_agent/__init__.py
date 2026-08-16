# my-agent research_agent 包入口（只导出主线模块）
# 2026-08-16: 移除 from .utils import * —— utils.py 792 行全部为死代码
# （除 rm.py import 后未实例化的 WebPageHelper），星号导入会把
# trafilatura/httpx/toml/tqdm 等重型依赖拖进包导入。
from .lm import *
from .rm import *

__version__ = "1.0.0"
