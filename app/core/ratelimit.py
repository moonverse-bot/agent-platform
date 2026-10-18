"""限流:令牌桶。

优先使用 Redis,保证多实例部署下计数一致;Redis 不可用时自动退化为进程内实现,
这样本地开发不装 Redis 也能跑。Redis 侧的「读取-补充-扣减-写回」用一段 Lua
保证原子性,避免并发下的超额放行。
"""

import logging
import time

logger = logging.getLogger(__name__)

_TOKEN_BUCKET_LUA = """
local key = KEYS[1]
local capacity = tonumber(ARGV[1])
local refill_per_sec = tonumber(ARGV[2])
local now_ms = tonumber(ARGV[3])
local cost = tonumber(ARGV[4])

local data = redis.call('HMGET', key, 'tokens', 'ts')
local tokens = tonumber(data[1])
local ts = tonumber(data[2])
if tokens == nil then
  tokens = capacity
  ts = now_ms
end

local elapsed = math.max(0, now_ms - ts) / 1000
tokens = math.min(capacity, tokens + elapsed * refill_per_sec)

local allowed = 0
if tokens >= cost then
  tokens = tokens - cost
  allowed = 1
end

redis.call('HMSET', key, 'tokens', tokens, 'ts', now_ms)
redis.call('PEXPIRE', key, math.ceil(capacity / refill_per_sec * 1000) + 1000)
return {allowed, math.floor(tokens)}
"""


class RateLimitResult:
    __slots__ = ("allowed", "remaining", "limit")

    def __init__(self, allowed: bool, remaining: int, limit: int) -> None:
        self.allowed = allowed
        self.remaining = remaining
        self.limit = limit


class InMemoryTokenBucket:
    """进程内令牌桶(单机部署 / 本地开发用)。"""

    def __init__(self, capacity: int, refill_per_sec: float) -> None:
        self._capacity = float(capacity)
        self._refill = refill_per_sec
        self._buckets: dict[str, list[float]] = {}

    async def consume(self, key: str, cost: float = 1.0) -> tuple[bool, int]:
        now = time.monotonic()
        tokens, last = self._buckets.get(key, [self._capacity, now])
        tokens = min(self._capacity, tokens + max(0.0, now - last) * self._refill)
        allowed = tokens >= cost
        if allowed:
            tokens -= cost
        self._buckets[key] = [tokens, now]
        return allowed, int(tokens)


class RedisTokenBucket:
    """Redis 令牌桶(多实例共享计数)。"""

    def __init__(self, client, capacity: int, refill_per_sec: float) -> None:
        self._capacity = capacity
        self._refill = refill_per_sec
        self._script = client.register_script(_TOKEN_BUCKET_LUA)

    async def consume(self, key: str, cost: float = 1.0) -> tuple[bool, int]:
        allowed, remaining = await self._script(
            keys=[key],
            args=[self._capacity, self._refill, int(time.time() * 1000), cost],
        )
        return bool(int(allowed)), int(remaining)


class RateLimiter:
    """对外只暴露 hit();具体后端在启动时决定。"""

    def __init__(self, per_minute: int, backend=None) -> None:
        self.limit = per_minute
        self.backend_name = "memory"
        self._backend = backend or InMemoryTokenBucket(per_minute, per_minute / 60.0)

    async def hit(self, key: str, cost: float = 1.0) -> RateLimitResult:
        allowed, remaining = await self._backend.consume(f"ratelimit:{key}", cost)
        return RateLimitResult(allowed, remaining, self.limit)


async def build_rate_limiter(per_minute: int, redis_url: str) -> RateLimiter:
    limiter = RateLimiter(per_minute)
    if not redis_url:
        logger.info("未配置 REDIS_URL,限流使用进程内实现", extra={"event": "ratelimit.memory"})
        return limiter

    try:
        from redis.asyncio import Redis

        client = Redis.from_url(redis_url, decode_responses=True)
        await client.ping()
    except Exception as exc:  # noqa: BLE001 - 任何连接问题都退化为进程内实现
        logger.warning(
            "Redis 不可用,限流退化为进程内实现: %s", exc, extra={"event": "ratelimit.fallback"}
        )
        return limiter

    limiter._backend = RedisTokenBucket(client, per_minute, per_minute / 60.0)  # noqa: SLF001
    limiter.backend_name = "redis"
    logger.info("限流使用 Redis 后端", extra={"event": "ratelimit.redis"})
    return limiter
