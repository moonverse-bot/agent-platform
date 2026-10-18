"""限流:令牌桶的基本行为。"""

import asyncio

from app.core.ratelimit import InMemoryTokenBucket, RateLimiter


async def test_bucket_allows_up_to_capacity_then_blocks():
    bucket = InMemoryTokenBucket(capacity=3, refill_per_sec=0.0)
    allowed = [(await bucket.consume("user:1"))[0] for _ in range(5)]
    assert allowed == [True, True, True, False, False]


async def test_bucket_refills_over_time():
    bucket = InMemoryTokenBucket(capacity=1, refill_per_sec=100.0)
    assert (await bucket.consume("user:1"))[0] is True
    # 100 令牌/秒 → 20ms 补 2 个,足够放行下一次
    await asyncio.sleep(0.02)
    assert (await bucket.consume("user:1"))[0] is True


async def test_keys_are_isolated():
    bucket = InMemoryTokenBucket(capacity=1, refill_per_sec=0.0)
    assert (await bucket.consume("user:1"))[0] is True
    assert (await bucket.consume("user:1"))[0] is False
    assert (await bucket.consume("user:2"))[0] is True


async def test_limiter_reports_limit_and_remaining():
    limiter = RateLimiter(per_minute=2, backend=InMemoryTokenBucket(2, 0.0))
    first = await limiter.hit("user:1")
    second = await limiter.hit("user:1")
    third = await limiter.hit("user:1")

    assert first.allowed and first.remaining == 1 and first.limit == 2
    assert second.allowed and second.remaining == 0
    assert third.allowed is False and third.remaining == 0
