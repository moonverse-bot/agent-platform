"""进程内熔断器:外部依赖出错时先断开一段时间,别让每次请求都去等一次超时。

刻意只做「开 / 关」两个状态,不做滑动窗口计数:调用方(知识库检索)只要知道
「现在还要不要试」就够了。滚动指标、半开试探这些留给跨进程的真实实现
(比如放 Redis),这里保持零依赖、好读、好测。

`clock` 可注入是为了测试能直接操控时间,而不必真的 sleep。
"""

import time
from collections.abc import Callable


class CircuitBreaker:
    def __init__(
        self,
        cooldown_seconds: float = 60.0,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._cooldown = cooldown_seconds
        self._clock = clock
        self._open_until = 0.0

    @property
    def is_open(self) -> bool:
        """熔断中(冷却还没结束)。调用方看到 True 就应该跳过这次外部调用。"""
        return self._clock() < self._open_until

    def trip(self) -> None:
        """外部调用失败后调用:从此刻起断开 cooldown_seconds 秒。"""
        self._open_until = self._clock() + self._cooldown

    def reset(self) -> None:
        """立刻恢复可用(测试用,也方便运维排查完马上开启)。"""
        self._open_until = 0.0
