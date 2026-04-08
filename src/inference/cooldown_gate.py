"""Cooldown gate to avoid spamming repeated predictions."""

from datetime import datetime, timedelta


class CooldownGate:
    """Allow outputs only after a minimum elapsed cooldown."""

    def __init__(self, cooldown_seconds: float = 0.5) -> None:
        self.cooldown = timedelta(seconds=cooldown_seconds)
        self.last_emit: datetime | None = None

    def allow(self) -> bool:
        """Return True when a new emit is allowed."""
        now = datetime.utcnow()
        if self.last_emit is None or (now - self.last_emit) >= self.cooldown:
            self.last_emit = now
            return True
        return False
