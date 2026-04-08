"""Basic tests for inference cooldown gate behavior."""

from src.inference.cooldown_gate import CooldownGate


def test_cooldown_gate_blocks_immediate_second_emit() -> None:
    gate = CooldownGate(cooldown_seconds=10.0)
    assert gate.allow() is True
    assert gate.allow() is False
