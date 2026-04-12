"""
OD (Origin-Destination) tracker.

Pairs boarding events with alighting events for each tracked passenger.
Emits completed OD records to the database when a passenger exits.

Double-counting prevention:
  - A track_id can only have ONE active boarding record at a time.
  - If FastReID resurrects a lost track_id, its existing boarding record
    continues instead of creating a duplicate.
  - Records orphaned by system shutdown are closed at route end.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from od.stop_manager import StopManager, GpsCoord
from tripwire.tripwire_manager import CrossingEvent, CrossingType

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class PassengerRecord:
    """In-flight OD record for a single passenger."""
    record_id: int                     # database primary key (0 = not yet persisted)
    track_id: int
    board_stop_id: str
    board_timestamp: datetime
    board_gps: GpsCoord
    alight_stop_id: Optional[str] = None
    alight_timestamp: Optional[datetime] = None
    alight_gps: Optional[GpsCoord] = None
    is_complete: bool = False


# ---------------------------------------------------------------------------
# ODTracker
# ---------------------------------------------------------------------------

class ODTracker:
    """
    Processes tripwire CrossingEvents and maintains the OD matrix.

    Dependencies injected at construction time so they can be swapped
    for unit testing.
    """

    def __init__(
        self,
        stop_manager: StopManager,
        db,                           # storage.db.Database (type-free to avoid circular import)
        route_id: str = "UNKNOWN",
        vehicle_id: str = "UNKNOWN",
    ) -> None:
        self.stop_manager = stop_manager
        self.db = db
        self.route_id = route_id
        self.vehicle_id = vehicle_id

        # track_id → active PassengerRecord (boarding not yet paired with exit)
        self._active: dict[int, PassengerRecord] = {}
        # Completed OD records (for in-memory analytics)
        self._completed: list[PassengerRecord] = []

    # ------------------------------------------------------------------
    def process_event(
        self,
        event: CrossingEvent,
        timestamp: datetime,
        current_gps: Optional[GpsCoord],
    ) -> Optional[PassengerRecord]:
        """
        Handle a single CrossingEvent from TripwireManager.

        Returns the completed PassengerRecord if an alighting event finishes
        a boarding record, otherwise None.
        """
        if event.crossing_type == CrossingType.BOARD:
            return self._on_board(event, timestamp, current_gps)
        else:
            return self._on_alight(event, timestamp, current_gps)

    # ------------------------------------------------------------------
    def _on_board(
        self,
        event: CrossingEvent,
        timestamp: datetime,
        gps: Optional[GpsCoord],
    ) -> None:
        track_id = event.track_id

        # Guard: prevent double-counting if track_id is already boarding
        if track_id in self._active:
            logger.debug(
                "Boarding ignored: track %d already has an active record", track_id
            )
            return None

        stop_id = (
            self.stop_manager.get_current_stop(gps) if gps else None
        ) or "UNKNOWN"

        record = PassengerRecord(
            record_id=0,
            track_id=track_id,
            board_stop_id=stop_id,
            board_timestamp=timestamp,
            board_gps=gps or (0.0, 0.0),
        )

        # Persist to DB and get assigned record_id
        record.record_id = self.db.insert_od_event(record, self.route_id, self.vehicle_id)
        self._active[track_id] = record

        logger.info(
            "BOARD  track=%d stop=%s ts=%s",
            track_id, stop_id, timestamp.isoformat()
        )
        return None

    def _on_alight(
        self,
        event: CrossingEvent,
        timestamp: datetime,
        gps: Optional[GpsCoord],
    ) -> Optional[PassengerRecord]:
        track_id = event.track_id
        record = self._active.pop(track_id, None)

        if record is None:
            logger.debug(
                "Alighting event for track %d with no boarding record (missed boarding?)",
                track_id,
            )
            return None

        stop_id = (
            self.stop_manager.get_current_stop(gps) if gps else None
        ) or "UNKNOWN"

        record.alight_stop_id = stop_id
        record.alight_timestamp = timestamp
        record.alight_gps = gps or (0.0, 0.0)
        record.is_complete = True

        # Update DB record
        self.db.mark_alight(
            record_id=record.record_id,
            alight_stop=stop_id,
            alight_ts=timestamp,
            alight_gps=record.alight_gps,
        )
        self._completed.append(record)

        logger.info(
            "ALIGHT track=%d  %s → %s",
            track_id, record.board_stop_id, stop_id
        )
        return record

    # ------------------------------------------------------------------
    def resurrect_track(self, old_track_id: int, new_track_id: int) -> None:
        """
        Called when FastReID re-identifies a lost passenger under a new
        ByteTrack track_id.

        Transfers the active boarding record from old_track_id to new_track_id
        so the passenger's OD pairing is preserved.
        """
        record = self._active.pop(old_track_id, None)
        if record is None:
            return
        record.track_id = new_track_id
        self._active[new_track_id] = record
        logger.info(
            "Re-ID resurrection: track %d → %d (stop=%s)",
            old_track_id, new_track_id, record.board_stop_id
        )

    def close_orphan_records(
        self,
        active_track_ids: set[int],
        timestamp: datetime,
        stop_id: str = "ROUTE_END",
    ) -> list[PassengerRecord]:
        """
        Close all boarding records that have no matching alighting event.
        Called at end-of-route or during graceful shutdown.

        Returns list of closed (orphaned) records.
        """
        orphans: list[PassengerRecord] = []
        orphan_ids = [tid for tid in self._active if tid not in active_track_ids]

        for track_id in orphan_ids:
            record = self._active.pop(track_id)
            record.alight_stop_id = stop_id
            record.alight_timestamp = timestamp
            record.is_complete = True
            self.db.mark_alight(
                record_id=record.record_id,
                alight_stop=stop_id,
                alight_ts=timestamp,
                alight_gps=record.board_gps,
            )
            self._completed.append(record)
            orphans.append(record)
            logger.warning(
                "Orphan record closed: track=%d boarded at %s",
                track_id, record.board_stop_id
            )

        return orphans

    # ------------------------------------------------------------------
    def get_od_matrix(self) -> dict[tuple[str, str], int]:
        """
        Return count of completed OD pairs.

        Returns:
            dict mapping (board_stop_id, alight_stop_id) → passenger count
        """
        matrix: dict[tuple[str, str], int] = {}
        for rec in self._completed:
            if rec.is_complete and rec.alight_stop_id:
                key = (rec.board_stop_id, rec.alight_stop_id)
                matrix[key] = matrix.get(key, 0) + 1
        return matrix

    def active_count(self) -> int:
        """Number of passengers currently on the bus (boarded but not alighted)."""
        return len(self._active)

    def total_completed(self) -> int:
        return len(self._completed)
