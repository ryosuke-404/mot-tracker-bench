"""
Unit tests for ODTracker: boarding/alighting pairing and double-count prevention.
Uses an in-memory SQLite database.
"""

import sys
import tempfile
from datetime import datetime
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from od.od_tracker import ODTracker
from od.stop_manager import StopManager
from storage.db import Database
from tripwire.tripwire_manager import CrossingEvent, CrossingType


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def db(tmp_path):
    return Database(str(tmp_path / "test.db"))


@pytest.fixture
def stop_mgr(tmp_path):
    import json
    stops = [
        {"stop_id": "S01", "name": "Stop A", "lat": 35.00, "lon": 135.00},
        {"stop_id": "S02", "name": "Stop B", "lat": 35.01, "lon": 135.01},
    ]
    stop_file = tmp_path / "stops.json"
    stop_file.write_text(json.dumps(stops))
    mgr = StopManager(str(stop_file), proximity_radius_m=200.0)
    mgr.load_stops()
    return mgr


@pytest.fixture
def tracker(db, stop_mgr):
    return ODTracker(stop_manager=stop_mgr, db=db, route_id="R1", vehicle_id="V1")


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def board_event(track_id: int) -> CrossingEvent:
    return CrossingEvent(
        track_id=track_id, door_id="front_door", camera_id="front",
        crossing_type=CrossingType.BOARD, frame_id=1, position=(0.5, 0.5),
    )


def alight_event(track_id: int) -> CrossingEvent:
    return CrossingEvent(
        track_id=track_id, door_id="front_door", camera_id="front",
        crossing_type=CrossingType.ALIGHT, frame_id=100, position=(0.5, 0.5),
    )


TS_BOARD = datetime(2024, 1, 1, 9, 0, 0)
TS_ALIGHT = datetime(2024, 1, 1, 9, 15, 0)
GPS_S01 = (35.001, 135.001)   # near stop S01
GPS_S02 = (35.011, 135.011)   # near stop S02


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_basic_od_pair(tracker):
    tracker.process_event(board_event(1), TS_BOARD, GPS_S01)
    assert tracker.active_count() == 1

    record = tracker.process_event(alight_event(1), TS_ALIGHT, GPS_S02)
    assert record is not None
    assert record.is_complete
    assert record.track_id == 1
    assert record.board_stop_id in ("S01", "UNKNOWN")
    assert tracker.active_count() == 0


def test_double_board_ignored(tracker):
    """Second boarding event for same track_id must be ignored."""
    tracker.process_event(board_event(1), TS_BOARD, GPS_S01)
    tracker.process_event(board_event(1), TS_BOARD, GPS_S01)  # duplicate
    assert tracker.active_count() == 1


def test_alight_without_board(tracker):
    """Alighting for a track with no boarding record should not crash."""
    result = tracker.process_event(alight_event(99), TS_ALIGHT, GPS_S02)
    assert result is None
    assert tracker.active_count() == 0


def test_reid_resurrection(tracker):
    """
    After Re-ID resurrection, boarding record transfers to new track_id.
    """
    tracker.process_event(board_event(1), TS_BOARD, GPS_S01)
    # ByteTrack assigned new id=5, Re-ID matched to old id=1
    tracker.resurrect_track(old_track_id=1, new_track_id=5)

    assert 1 not in tracker._active
    assert 5 in tracker._active
    assert tracker._active[5].board_stop_id is not None

    # Alighting under new id=5 should complete the OD record
    record = tracker.process_event(alight_event(5), TS_ALIGHT, GPS_S02)
    assert record is not None
    assert record.is_complete


def test_od_matrix(tracker):
    tracker.process_event(board_event(1), TS_BOARD, GPS_S01)
    tracker.process_event(alight_event(1), TS_ALIGHT, GPS_S02)

    tracker.process_event(board_event(2), TS_BOARD, GPS_S01)
    tracker.process_event(alight_event(2), TS_ALIGHT, GPS_S02)

    matrix = tracker.get_od_matrix()
    assert tracker.total_completed() == 2


def test_orphan_close(tracker):
    """close_orphan_records should close any boarding without alighting."""
    tracker.process_event(board_event(10), TS_BOARD, GPS_S01)
    tracker.process_event(board_event(11), TS_BOARD, GPS_S01)
    assert tracker.active_count() == 2

    orphans = tracker.close_orphan_records(
        active_track_ids=set(), timestamp=TS_ALIGHT, stop_id="END"
    )
    assert len(orphans) == 2
    assert tracker.active_count() == 0
