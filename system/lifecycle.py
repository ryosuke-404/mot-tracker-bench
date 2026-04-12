"""
System lifecycle management for bus edge deployment.

Handles:
  - Graceful shutdown on SIGTERM / SIGINT
  - Ignition-controlled power management (ACC signal via GPIO)
  - Watchdog for pipeline health
  - Safe shutdown with data flush before power cut
"""

from __future__ import annotations

import logging
import signal
import threading
import time
from typing import Callable, Optional

logger = logging.getLogger(__name__)


class SystemLifecycle:
    """
    Manages startup, shutdown, and watchdog for the bus tracking system.

    The ignition GPIO pin (ACC line) controls automatic start/stop:
      - Ignition ON  → start tracking pipeline
      - Ignition OFF → flush data → safe shutdown after delay_s

    On hardware without GPIO (development), set `ignition_gpio_pin=None`.
    """

    def __init__(
        self,
        ignition_gpio_pin: Optional[int] = None,
        shutdown_delay_s: float = 10.0,
        watchdog_interval_s: float = 5.0,
    ) -> None:
        self.ignition_gpio_pin = ignition_gpio_pin
        self.shutdown_delay_s = shutdown_delay_s
        self.watchdog_interval_s = watchdog_interval_s

        self._shutdown_event = threading.Event()
        self._shutdown_callbacks: list[Callable] = []
        self._last_heartbeat: float = time.monotonic()
        self._watchdog_thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------
    def setup_signal_handlers(self) -> None:
        """Register SIGTERM / SIGINT for graceful shutdown."""
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)
        logger.info("Signal handlers registered (SIGTERM, SIGINT)")

    def _signal_handler(self, signum, frame) -> None:
        logger.info("Signal %d received — initiating graceful shutdown", signum)
        self._initiate_shutdown()

    # ------------------------------------------------------------------
    def on_shutdown(self, callback: Callable) -> None:
        """Register a callback to be invoked during shutdown."""
        self._shutdown_callbacks.append(callback)

    def wait_for_shutdown(self) -> None:
        """Block until shutdown is triggered."""
        self._shutdown_event.wait()

    @property
    def is_running(self) -> bool:
        return not self._shutdown_event.is_set()

    # ------------------------------------------------------------------
    def wait_for_ignition(self) -> bool:
        """
        Block until the ignition ACC pin goes HIGH.

        Returns True when ignition is detected, False if GPIO unavailable
        (falls through immediately in development mode).
        """
        if self.ignition_gpio_pin is None:
            logger.info("No GPIO pin configured — skipping ignition wait (dev mode)")
            return False

        try:
            import Jetson.GPIO as GPIO
        except ImportError:
            logger.warning("Jetson.GPIO not available — skipping ignition wait")
            return False

        GPIO.setmode(GPIO.BOARD)
        GPIO.setup(self.ignition_gpio_pin, GPIO.IN)
        logger.info("Waiting for ignition signal on GPIO pin %d …", self.ignition_gpio_pin)

        while True:
            if GPIO.input(self.ignition_gpio_pin) == GPIO.HIGH:
                logger.info("Ignition ON detected")
                return True
            time.sleep(0.5)

    def monitor_ignition_off(self) -> None:
        """
        Spawn a background thread that monitors the ACC pin for
        ignition OFF and triggers graceful shutdown.
        """
        if self.ignition_gpio_pin is None:
            return

        def _monitor():
            try:
                import Jetson.GPIO as GPIO
            except ImportError:
                return

            while not self._shutdown_event.is_set():
                if GPIO.input(self.ignition_gpio_pin) == GPIO.LOW:
                    logger.info(
                        "Ignition OFF detected — shutdown in %.1fs",
                        self.shutdown_delay_s
                    )
                    time.sleep(self.shutdown_delay_s)
                    self._initiate_shutdown()
                    return
                time.sleep(1.0)

        t = threading.Thread(target=_monitor, daemon=True, name="IgnitionMonitor")
        t.start()

    # ------------------------------------------------------------------
    def heartbeat(self) -> None:
        """Call this from the pipeline loop to signal liveness."""
        self._last_heartbeat = time.monotonic()

    def start_watchdog(self) -> None:
        """Start watchdog thread; triggers shutdown if heartbeat stops."""
        def _watchdog():
            time.sleep(self.watchdog_interval_s * 2)  # grace period at startup
            while not self._shutdown_event.is_set():
                elapsed = time.monotonic() - self._last_heartbeat
                if elapsed > self.watchdog_interval_s * 3:
                    logger.error(
                        "Watchdog: no heartbeat for %.1fs — triggering restart",
                        elapsed
                    )
                    self._initiate_shutdown()
                    return
                time.sleep(self.watchdog_interval_s)

        self._watchdog_thread = threading.Thread(
            target=_watchdog, daemon=True, name="Watchdog"
        )
        self._watchdog_thread.start()
        logger.info("Watchdog started (interval=%.1fs)", self.watchdog_interval_s)

    # ------------------------------------------------------------------
    def _initiate_shutdown(self) -> None:
        if self._shutdown_event.is_set():
            return  # already shutting down
        logger.info("Graceful shutdown initiated")
        for cb in self._shutdown_callbacks:
            try:
                cb()
            except Exception as exc:
                logger.exception("Shutdown callback error: %s", exc)
        self._shutdown_event.set()
