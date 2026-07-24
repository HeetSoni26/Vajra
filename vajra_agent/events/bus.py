"""EventBus architecture for publishing and subscribing to AgentEvents."""

from __future__ import annotations

from collections.abc import Callable

from vajra_agent.events.types import AgentEvent

EventListener = Callable[[AgentEvent], None]


class EventBus:
    """Lightweight event dispatcher allowing subscribers to monitor agent execution."""

    def __init__(self) -> None:
        self._listeners: list[EventListener] = []

    def subscribe(self, listener: EventListener) -> None:
        """Register a new event listener callback."""
        if listener not in self._listeners:
            self._listeners.append(listener)

    def unsubscribe(self, listener: EventListener) -> None:
        """Unregister an event listener callback."""
        if listener in self._listeners:
            self._listeners.remove(listener)

    def emit(self, event: AgentEvent) -> None:
        """Emit an event to all subscribed listeners."""
        for listener in self._listeners:
            try:
                listener(event)
            except Exception:
                # Event listener exceptions must not crash agent execution
                pass
