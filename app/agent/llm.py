"""大模型适配层:统一走 OpenAI 兼容协议,换供应商只改配置。"""

import logging

from openai import AsyncOpenAI

from app.config import Settings

logger = logging.getLogger(__name__)


def build_client(settings: Settings) -> AsyncOpenAI | None:
    """没有配置 API Key 时返回 None,由调用方决定降级行为。"""
    api_key = settings.llm_api_key.strip()
    base_url = settings.llm_base_url.strip()
    if not api_key:
        logger.warning("未配置 DEEPSEEK_API_KEY,Agent 不可用", extra={"event": "llm.disabled"})
        return None

    # 环境变量里经常混入复制粘贴带来的首尾空白,会让请求头变成非法值,
    # 报错却只显示成连接失败,极难排查,这里统一清掉。
    return AsyncOpenAI(
        api_key=api_key,
        base_url=base_url or "https://api.deepseek.com",
        timeout=settings.llm_timeout_seconds,
        max_retries=2,
    )
