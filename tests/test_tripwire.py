"""
Unit tests for virtual tripwire manager.
No GPU or camera required.
"""

import sys
import tempfile
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tripwire.tripwire_manager import TripwireManager, CrossingType


# ---------------------------------------------------------------------------
# Config fixture
# ---------------------------------------------------------------------------

_TRIPWIRE_YAML = """
cameras:
  front:
    doors:
      - door_id: front_door
        lines:
          outer:
            line_id: front_door_outer
            p1: [0.30, 0.0]
            p2: [0.30, 1.0]
            inward_normal: [1.0, 0.0]
          inner:
            line_id: front_door_inner
            p1: [0.60, 0.0]
            p2: [0.60, 1.0]
            inward_normal: [1.0, 0.0]
"""


@pytest.fixture
def manager():
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False
    ) as f:
        f.write(_TRIPWIRE_YAML)
        config_path = f.name

    mgr = TripwireManager(
        config_path=config_path,
        sequence_timeout_frames=50,
        direction_dot_thresh=0.2,
    )
    mgr.load_config()
    return mgr


# ---------------------------------------------------------------------------
# Boarding: outer inward → inner inward
# ---------------------------------------------------------------------------

def test_board_event(manager):
    """
    Track moves from x=0.2 → x=0.35 → x=0.65 (crosses outer then inner, inward).
    Expected: one BOARD event.
    """
    events_total = []

    # Frame 1: cross outer (0.2→0.35 crossing the 0.30 line, inward direction)
    ev = manager.update(
        camera_id="front",
        frame_id=1,
        track_positions={1: (0.35, 0.5)},
        prev_positions={1: (0.20, 0.5)},
    )
    events_total.extend(ev)

    # Frame 2: cross inner (0.35→0.65 crossing the 0.60 line, inward direction)
    ev = manager.update(
        camera_id="front",
        frame_id=2,
        track_positions={1: (0.65, 0.5)},
        prev_positions={1: (0.35, 0.5)},
    )
    events_total.extend(ev)

    boards = [e for e in events_total if e.crossing_type == CrossingType.BOARD]
    assert len(boards) == 1
    assert boards[0].track_id == 1


def test_alight_event(manager):
    """
    Track moves from x=0.8 → x=0.55 → x=0.25 (crosses inner then outer, outward).
    Expected: one ALIGHT event.
    """
    events_total = []

    # Cross inner outward (0.8 → 0.55)
    ev = manager.update(
        camera_id="front",
        frame_id=10,
        track_positions={2: (0.55, 0.5)},
        prev_positions={2: (0.80, 0.5)},
    )
    events_total.extend(ev)

    # Cross outer outward (0.55 → 0.25)
    ev = manager.update(
        camera_id="front",
        frame_id=11,
        track_positions={2: (0.25, 0.5)},
        prev_positions={2: (0.55, 0.5)},
    )
    events_total.extend(ev)

    alights = [e for e in events_total if e.crossing_type == CrossingType.ALIGHT]
    assert len(alights) == 1
    assert alights[0].track_id == 2


def test_no_event_for_jitter(manager):
    """Small oscillation that never crosses both lines should produce no event."""
    events = []
    for frame_id in range(20):
        x = 0.28 if frame_id % 2 == 0 else 0.32
        ev = manager.update(
            camera_id="front",
            frame_id=frame_id,
            track_positions={3: (x, 0.5)},
            prev_positions={3: (0.32 if frame_id % 2 == 0 else 0.28, 0.5)},
        )
        events.extend(ev)

    assert len(events) == 0, f"Expected no events for jitter, got: {events}"


def test_sequence_timeout_resets(manager):
    """
    If the inner crossing doesn't happen within sequence_timeout_frames,
    a second outer crossing should restart the sequence and eventually
    produce ONE board event.
    """
    events = []

    # Cross outer inward at frame 1
    manager.update(
        camera_id="front", frame_id=1,
        track_positions={4: (0.35, 0.5)}, prev_positions={4: (0.20, 0.5)},
    )

    # Jump past timeout (> 50 frames) without inner crossing
    # Cross outer inward again at frame 60
    ev = manager.update(
        camera_id="front", frame_id=60,
        track_positions={4: (0.35, 0.5)}, prev_positions={4: (0.20, 0.5)},
    )
    events.extend(ev)

    # Now cross inner inward at frame 61
    ev = manager.update(
        camera_id="front", frame_id=61,
        track_positions={4: (0.65, 0.5)}, prev_positions={4: (0.35, 0.5)},
    )
    events.extend(ev)

    boards = [e for e in events if e.crossing_type == CrossingType.BOARD]
    assert len(boards) == 1
