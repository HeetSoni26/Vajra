"""
Watchdog thread for Vajra production training.
Detects freezes, stalled dataloaders, and unexpected inactivity.
"""

from __future__ import annotations

import threading
import time
from typing import Callable

from utils.logging import setup_logger

logger = setup_logger("watchdog")


class Watchdog:
    """
    Background thread that monitors training heartbeats.
    If no heartbeat is received within `timeout_seconds`, it fires the `on_trigger` callback.

    Typical use: emergency checkpoint save + graceful shutdown.
    """

    def __init__(
        self,
        timeout_seconds: float = 300.0,
        on_trigger: Callable[[], None] | None = None,
        check_interval: float = 10.0,
    ) -> None:
        self.timeout_seconds = timeout_seconds
        self.on_trigger = on_trigger
        self.check_interval = check_interval
        self._last_heartbeat = time.time()
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True, name="vajra-watchdog")

    def start(self) -> None:
        self._thread.start()
        logger.info(f"Watchdog started (timeout={self.timeout_seconds}s).")

    def stop(self) -> None:
        self._stop_event.set()

    def heartbeat(self) -> None:
        """Call at each training step to prevent watchdog trigger."""
        self._last_heartbeat = time.time()

    def _run(self) -> None:
        while not self._stop_event.is_set():
            time.sleep(self.check_interval)
            elapsed = time.time() - self._last_heartbeat
            if elapsed > self.timeout_seconds:
                logger.error(
                    f"[WATCHDOG] Training appears frozen — no heartbeat for {elapsed:.0f}s "
                    f"(timeout={self.timeout_seconds}s). Triggering emergency recovery."
                )
                if self.on_trigger:
                    try:
                        self.on_trigger()
                    except Exception as e:
                        logger.error(f"[WATCHDOG] Emergency callback failed: {e}")
                # Reset so we don't fire continuously
                self._last_heartbeat = time.time()
