"""Unit tests for AgentLoop completion detection and stop conditions."""

from vllm_ascend_harness.core.agent_loop import (
    FINISH_SENTINELS,
    AgentContext,
    AgentLoop,
    AgentStep,
)
from vllm_ascend_harness.core.event_bus import EventBus


class SimpleAgentLoop(AgentLoop):
    """Test loop that echoes input and stops on sentinel."""

    def __init__(self, context, event_bus=None, outputs=None, mark_finished=False):
        super().__init__(context, event_bus)
        self._outputs = outputs or []
        self._call_count = 0
        self._mark_finished = mark_finished

    def step(self) -> AgentStep:
        if self._call_count < len(self._outputs):
            output = self._outputs[self._call_count]
            is_last = self._call_count == len(self._outputs) - 1
        else:
            output = "[DONE]"
            is_last = True
        self._call_count += 1
        finished = self._mark_finished and is_last
        return AgentStep(skill="test", input="n/a", output=output, finished=finished)


class TestAgentStep:
    def test_default_finished_false(self):
        step = AgentStep(skill="x", input="y")
        assert step.finished is False

    def test_finished_can_be_set(self):
        step = AgentStep(skill="x", input="y", finished=True)
        assert step.finished is True


class TestAgentContext:
    def test_defaults(self):
        ctx = AgentContext(task_id="t1", prompt="hello")
        assert ctx.steps == []
        assert ctx.max_steps == 20
        assert ctx.metadata == {}


class TestAgentLoopRun:
    def test_run_stops_on_finished_flag(self):
        ctx = AgentContext(task_id="t1", prompt="p", max_steps=10)
        loop = SimpleAgentLoop(ctx, outputs=["result"], mark_finished=True)
        loop.run()
        assert len(ctx.steps) == 1
        assert ctx.steps[0].finished is True

    def test_run_stops_on_sentinel(self):
        ctx = AgentContext(task_id="t1", prompt="p", max_steps=10)
        loop = SimpleAgentLoop(ctx, outputs=["[DONE]"])
        loop.run()
        assert len(ctx.steps) == 1

    def test_run_stops_on_finish_sentinel(self):
        ctx = AgentContext(task_id="t1", prompt="p", max_steps=10)
        loop = SimpleAgentLoop(ctx, outputs=["[FINISH]"])
        loop.run()
        assert len(ctx.steps) == 1

    def test_run_stops_on_end_tag(self):
        ctx = AgentContext(task_id="t1", prompt="p", max_steps=10)
        loop = SimpleAgentLoop(ctx, outputs=["<task_complete>"])
        loop.run()
        assert len(ctx.steps) == 1

    def test_run_stops_on_max_steps(self):
        ctx = AgentContext(task_id="t1", prompt="p", max_steps=3)
        loop = SimpleAgentLoop(ctx, outputs=["a", "b", "c", "d", "e"])
        loop.run()
        assert len(ctx.steps) == 3

    def test_run_publishes_events(self):
        received = []
        bus = EventBus()
        bus.subscribe("agent.step", received.append)
        ctx = AgentContext(task_id="t1", prompt="p", max_steps=5)
        loop = SimpleAgentLoop(ctx, event_bus=bus, outputs=["[DONE]"])
        loop.run()
        assert len(received) == 1
        assert isinstance(received[0], AgentStep)

    def test_run_empty_outputs_stops_on_done(self):
        ctx = AgentContext(task_id="t1", prompt="p", max_steps=10)
        loop = SimpleAgentLoop(ctx, outputs=[])
        loop.run()
        assert len(ctx.steps) == 1


class TestIsDone:
    def test_finished_flag(self):
        ctx = AgentContext(task_id="t1", prompt="p")
        loop = SimpleAgentLoop(ctx)
        step = AgentStep(skill="x", input="y", finished=True)
        assert loop._is_done(step) is True

    def test_sentinel_in_output(self):
        ctx = AgentContext(task_id="t1", prompt="p")
        loop = SimpleAgentLoop(ctx)
        for sentinel in FINISH_SENTINELS:
            step = AgentStep(skill="x", input="y", output=f"work {sentinel}")
            assert loop._is_done(step) is True

    def test_finish_reason_stop(self):
        ctx = AgentContext(task_id="t1", prompt="p")
        loop = SimpleAgentLoop(ctx)
        step = AgentStep(
            skill="x", input="y", output="ok", metadata={"finish_reason": "stop"}
        )
        assert loop._is_done(step) is True

    def test_finish_reason_length_not_done(self):
        ctx = AgentContext(task_id="t1", prompt="p")
        loop = SimpleAgentLoop(ctx)
        step = AgentStep(
            skill="x", input="y", output="ok", metadata={"finish_reason": "length"}
        )
        assert loop._is_done(step) is False

    def test_non_string_output_not_done(self):
        ctx = AgentContext(task_id="t1", prompt="p")
        loop = SimpleAgentLoop(ctx)
        step = AgentStep(skill="x", input="y", output=42)
        assert loop._is_done(step) is False
