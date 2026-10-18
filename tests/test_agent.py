"""Agent 循环:用假模型客户端覆盖工具调用、异常与降级,全程不联网。"""

from types import SimpleNamespace

from app.agent.runner import AgentRunner
from app.agent.tools import build_default_registry
from app.agent.tools.base import ToolContext
from app.config import Settings


def _chunk(content=None, tool_calls=None):
    delta = SimpleNamespace(content=content, tool_calls=tool_calls)
    return SimpleNamespace(choices=[SimpleNamespace(delta=delta, index=0)])


def _tool_call(index, id=None, name=None, arguments=None):
    return SimpleNamespace(
        index=index, id=id, function=SimpleNamespace(name=name, arguments=arguments)
    )


class _FakeStream:
    def __init__(self, chunks):
        self._chunks = list(chunks)

    def __aiter__(self):
        self._iter = iter(self._chunks)
        return self

    async def __anext__(self):
        try:
            return next(self._iter)
        except StopIteration:
            raise StopAsyncIteration


class _FakeCompletions:
    def __init__(self, rounds):
        self._rounds = list(rounds)
        self.calls: list[dict] = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return _FakeStream(self._rounds.pop(0))


class FakeClient:
    def __init__(self, rounds):
        self.chat = SimpleNamespace(completions=_FakeCompletions(rounds))

    @property
    def calls(self):
        return self.chat.completions.calls


def _runner(rounds, **overrides):
    settings = Settings(llm_api_key="test-key", **overrides)
    return AgentRunner(settings, client=FakeClient(rounds)), settings


async def _collect(runner, ctx: ToolContext, text="你好"):
    return [
        event
        async for event in runner.stream(
            session=ctx.session,
            user_id=ctx.user_id,
            conversation_id=None,
            history=[],
            user_text=text,
        )
    ]


async def test_plain_answer_streams_tokens(db_user: ToolContext):
    runner, _ = _runner([[_chunk(content="你好"), _chunk(content=",有什么可以帮你?")]])
    events = await _collect(runner, db_user)

    assert "".join(e.data["text"] for e in events if e.type == "token") == "你好,有什么可以帮你?"
    assert events[-1].type == "done"
    assert events[-1].data["content"] == "你好,有什么可以帮你?"


async def test_streaming_tool_arguments_are_reassembled(db_user: ToolContext):
    """工具参数是分片下发的,必须拼回完整 JSON 才能执行。"""
    rounds = [
        [
            _chunk(tool_calls=[_tool_call(0, id="call_1", name="search_", arguments='{"query": "')]),
            _chunk(tool_calls=[_tool_call(0, name="knowledge", arguments='退款政策"}')]),
        ],
        [_chunk(content="7 天内支持无理由退款。")],
    ]
    runner, _ = _runner(rounds)
    events = await _collect(runner, db_user, text="退款政策是什么")

    tool_events = [e for e in events if e.type == "tool"]
    assert len(tool_events) == 1
    assert tool_events[0].data["name"] == "search_knowledge"
    assert tool_events[0].data["arguments"] == {"query": "退款政策"}
    assert tool_events[0].data["ok"] is True
    assert "退款" in tool_events[0].data["result"]

    # 第二轮必须把工具结果作为 tool 消息带回模型
    second_round_messages = runner._client.chat.completions.calls[1]["messages"]
    assert any(m.get("role") == "tool" for m in second_round_messages)
    assert events[-1].data["content"] == "7 天内支持无理由退款。"


async def test_preamble_before_tool_call_is_discarded(db_user: ToolContext):
    """模型先吐一句铺垫再调工具时,必须下发 reset 让前端丢掉,避免界面与落库内容不一致。"""
    rounds = [
        [
            _chunk(content="I'll check your tickets."),
            _chunk(tool_calls=[_tool_call(0, id="c1", name="list_my_tickets", arguments="{}")]),
        ],
        [_chunk(content="你目前有 1 条工单。")],
    ]
    runner, _ = _runner(rounds)
    events = await _collect(runner, db_user)

    kinds = [e.type for e in events]
    assert "reset" in kinds
    # reset 必须出现在 tool 之前,前端才有机会在渲染工具卡片前清空文本
    assert kinds.index("reset") < kinds.index("tool")

    reset = next(e for e in events if e.type == "reset")
    assert reset.data["reason"] == "tool_call"
    assert "check" in reset.data["discarded"]

    # 最终下发给前端和落库的内容只包含真正的回答
    assert events[-1].data["content"] == "你目前有 1 条工单。"


async def test_no_reset_when_round_has_only_text(db_user: ToolContext):
    """没有工具调用的普通回答不应该触发 reset。"""
    runner, _ = _runner([[_chunk(content="你好")]])
    events = await _collect(runner, db_user)
    assert all(e.type != "reset" for e in events)


async def test_unknown_tool_is_reported_back_to_model(db_user: ToolContext):
    rounds = [
        [_chunk(tool_calls=[_tool_call(0, id="c1", name="shutdown_server", arguments="{}")])],
        [_chunk(content="我没有这个权限,已为你转人工。")],
    ]
    runner, _ = _runner(rounds)
    events = await _collect(runner, db_user)

    tool_event = next(e for e in events if e.type == "tool")
    assert tool_event.data["ok"] is False
    assert "不存在" in tool_event.data["result"]


async def test_invalid_tool_arguments_do_not_break_the_stream(db_user: ToolContext):
    rounds = [
        [_chunk(tool_calls=[_tool_call(0, id="c1", name="list_my_tickets", arguments="{不是JSON")])],
        [_chunk(content="刚刚没查成功,我再试一次。")],
    ]
    runner, _ = _runner(rounds)
    events = await _collect(runner, db_user)

    tool_event = next(e for e in events if e.type == "tool")
    assert tool_event.data["ok"] is False
    assert events[-1].type == "done"
    assert events[-1].data["content"] == "刚刚没查成功,我再试一次。"


async def test_tool_rounds_are_capped(db_user: ToolContext):
    """模型反复要调工具时,必须在配置的上限处停下,避免无限循环烧 token。"""
    rounds = [
        [_chunk(tool_calls=[_tool_call(0, id="c1", name="list_my_tickets", arguments="{}")])],
        [_chunk(tool_calls=[_tool_call(0, id="c2", name="list_my_tickets", arguments="{}")])],
    ]
    runner, _ = _runner(rounds, llm_max_tool_rounds=2)
    events = await _collect(runner, db_user)

    assert len([e for e in events if e.type == "tool"]) == 2
    assert "最大工具调用轮数" in events[-1].data["content"]


async def test_reasoner_model_does_not_send_temperature(db_user: ToolContext):
    runner, _ = _runner([[_chunk(content="ok")]], llm_model="deepseek-reasoner")
    await _collect(runner, db_user)
    assert "temperature" not in runner._client.chat.completions.calls[0]


async def test_missing_api_key_yields_actionable_error(db_user: ToolContext):
    runner = AgentRunner(
        Settings(llm_api_key="", llm_base_url="https://api.deepseek.com"),
        client=None,
        registry=build_default_registry(),
    )
    events = await _collect(runner, db_user)

    assert events[0].type == "error"
    assert "DEEPSEEK_API_KEY" in events[0].data["message"]
