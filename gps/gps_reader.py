"""
GPS reader: reads NMEA sentences from a serial port and maintains
the latest GPS fix in a thread-safe way.

Used by the pipeline to determine which bus stop is currently active.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Optional

logger = logging.getLogger(__name__)

GpsCoord = tuple[float, float]   # (latitude, longitude)


class GpsReader:
    """
    Background thread that reads NMEA $GPRMC / $GNRMC sentences from a
    serial port and exposes the latest GPS coordinate.
    """

    def __init__(self, port: str, baud_rate: int = 9600) -> None:
        self.port = port
        self.baud_rate = baud_rate

        self._latest_coord: Optional[GpsCoord] = None
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._running = False

    # ------------------------------------------------------------------
    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True, name="GPSReader")
        self._thread.start()
        logger.info("GPS reader started on %s @ %d baud", self.port, self.baud_rate)

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)

    # ------------------------------------------------------------------
    @property
    def latest_coord(self) -> Optional[GpsCoord]:
        """Return most recent (lat, lon) or None if no fix yet."""
        with self._lock:
            return self._latest_coord

    # ------------------------------------------------------------------
    def _run(self) -> None:
        try:
            import serial
        except ImportError:
            logger.error("pyserial is not installed. GPS reading disabled.")
            return

        while self._running:
            try:
                with serial.Serial(self.port, self.baud_rate, timeout=2.0) as ser:
                    logger.info("Serial port %s open", self.port)
                    while self._running:
                        line = ser.readline().decode("ascii", errors="ignore").strip()
                        coord = self._parse_nmea(line)
                        if coord:
                            with self._lock:
                                self._latest_coord = coord
            except Exception as exc:
                logger.warning("GPS serial error: %s — retrying in 5s", exc)
                time.sleep(5.0)

    # ------------------------------------------------------------------
    @staticmethod
    def _parse_nmea(sentence: str) -> Optional[GpsCoord]:
        """
        Parse a $GPRMC or $GNRMC sentence and return (lat, lon).

        $GPRMC,HHMMSS.ss,A,LLLL.LL,a,YYYYY.YY,a,x.x,x.x,DDMMYY,...*hh
        Field 2: status (A = active, V = void)
        Field 3: latitude DDMM.mmm
        Field 4: N or S
        Field 5: longitude DDDMM.mmm
        Field 6: E or W
        """
        if not sentence.startswith(("$GPRMC", "$GNRMC")):
            return None

        parts = sentence.split(",")
        if len(parts) < 7:
            return None
        if parts[2] != "A":   # not an active fix
            return None

        try:
            lat = _nmea_to_decimal(parts[3], parts[4])
            lon = _nmea_to_decimal(parts[5], parts[6].split("*")[0])
        except (ValueError, IndexError):
            return None

        return lat, lon


def _nmea_to_decimal(value: str, direction: str) -> float:
    """Convert NMEA DDDMM.mmm + N/S/E/W to decimal degrees."""
    if not value:
        raise ValueError("Empty NMEA value")
    dot = value.index(".")
    degrees = float(value[: dot - 2])
    minutes = float(value[dot - 2 :])
    decimal = degrees + minutes / 60.0
    if direction in ("S", "W"):
        decimal = -decimal
    return decimal


# ---------------------------------------------------------------------------
# Stub for development/testing without physical GPS
# ---------------------------------------------------------------------------

class MockGpsReader:
    """Returns a fixed GPS coordinate for offline development."""

    def __init__(self, coord: GpsCoord = (35.0167, 135.7556)) -> None:
        self._coord = coord

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    @property
    def latest_coord(self) -> GpsCoord:
        return self._coord
