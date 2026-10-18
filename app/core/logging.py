"""结构化日志:统一输出 JSON,并给每个请求串上 trace_id。"""

import json
import logging
import sys
from contextvars import ContextVar

trace_id_var: ContextVar[str] = ContextVar("trace_id", default="-")

_EXTRA_FIELDS = ("event", "latency_ms", "tool", "ok", "model", "tokens", "status_code")

# 按前缀匹配,而不是写死包名:httpx 的新版本把包名改成了 httpx2/httpcore2,
# 写死 "httpx" 就压不住它的 INFO 日志,每个请求都会刷一行非 JSON 的噪声。
_NOISY_PREFIXES = ("httpx", "httpcore", "openai", "uvicorn.access")


def set_trace_id(trace_id: str) -> None:
    trace_id_var.set(trace_id)


def get_trace_id() -> str:
    return trace_id_var.get()


class JsonFormatter(logging.Formatter):
    """把日志记录序列化成单行 JSON,方便被采集或检索。"""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "trace_id": trace_id_var.get(),
            "message": record.getMessage(),
        }
        for key in _EXTRA_FIELDS:
            value = getattr(record, key, None)
            if value is not None:
                payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def setup_logging(debug: bool = False) -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())

    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(logging.DEBUG if debug else logging.INFO)

    if not debug:
        _silence_noisy_loggers()


def _silence_noisy_loggers() -> None:
    """把 SDK 与 uvicorn 访问日志压到 WARNING,只留我们自己的结构化日志。"""
    for name, logger_object in list(logging.root.manager.loggerDict.items()):
        if isinstance(logger_object, logging.Logger) and name.startswith(_NOISY_PREFIXES):
            logger_object.setLevel(logging.WARNING)
