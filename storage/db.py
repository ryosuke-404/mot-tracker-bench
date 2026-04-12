"""
SQLite storage layer.

Schema stores only metadata (no video), minimising privacy risk and
bandwidth when syncing to cloud.

Tables:
  od_events    — completed or in-flight boarding/alighting pairs
  sync_queue   — records pending cloud upload
  system_log   — runtime logs for diagnostics
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from datetime import datetime
from typing import Any, Optional

logger = logging.getLogger(__name__)

GpsCoord = tuple[float, float]


class Database:
    """
    Thread-safe SQLite database for OD event storage.

    All writes go through a queue processed by a single writer thread to
    avoid blocking the tracking pipeline.
    """

    _CREATE_SQL = """
    CREATE TABLE IF NOT EXISTS od_events (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        route_id        TEXT    NOT NULL,
        vehicle_id      TEXT    NOT NULL,
        track_id        INTEGER NOT NULL,
        board_stop      TEXT    NOT NULL,
        board_ts        TEXT    NOT NULL,
        board_gps_lat   REAL,
        board_gps_lon   REAL,
        alight_stop     TEXT,
        alight_ts       TEXT,
        alight_gps_lat  REAL,
        alight_gps_lon  REAL,
        is_complete     INTEGER DEFAULT 0,
        created_at      TEXT    NOT NULL
    );

    CREATE TABLE IF NOT EXISTS sync_queue (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        od_event_id INTEGER NOT NULL REFERENCES od_events(id),
        payload     TEXT    NOT NULL,
        synced_at   TEXT
    );

    CREATE TABLE IF NOT EXISTS system_log (
        id      INTEGER PRIMARY KEY AUTOINCREMENT,
        ts      TEXT NOT NULL,
        level   TEXT NOT NULL,
        message TEXT NOT NULL
    );
    """

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        self._local = threading.local()   # thread-local connection
        self._write_lock = threading.Lock()
        self._init_db()

    # ------------------------------------------------------------------
    def _conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = sqlite3.connect(
                self.db_path,
                check_same_thread=False,
                detect_types=sqlite3.PARSE_DECLTYPES,
            )
            self._local.conn.row_factory = sqlite3.Row
            self._local.conn.execute("PRAGMA journal_mode=WAL")
            self._local.conn.execute("PRAGMA synchronous=NORMAL")
        return self._local.conn

    def _init_db(self) -> None:
        with self._write_lock:
            conn = self._conn()
            conn.executescript(self._CREATE_SQL)
            conn.commit()
        logger.info("Database initialised at %s", self.db_path)

    # ------------------------------------------------------------------
    def insert_od_event(
        self,
        record,                     # PassengerRecord (avoid circular import)
        route_id: str,
        vehicle_id: str,
    ) -> int:
        """
        Insert a boarding record (incomplete OD event).
        Returns the auto-assigned database ID.
        """
        now = datetime.utcnow().isoformat()
        with self._write_lock:
            conn = self._conn()
            cur = conn.execute(
                """
                INSERT INTO od_events
                    (route_id, vehicle_id, track_id, board_stop, board_ts,
                     board_gps_lat, board_gps_lon, is_complete, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, 0, ?)
                """,
                (
                    route_id,
                    vehicle_id,
                    record.track_id,
                    record.board_stop_id,
                    record.board_timestamp.isoformat(),
                    record.board_gps[0],
                    record.board_gps[1],
                    now,
                ),
            )
            conn.commit()
            return cur.lastrowid

    # ------------------------------------------------------------------
    def mark_alight(
        self,
        record_id: int,
        alight_stop: str,
        alight_ts: datetime,
        alight_gps: GpsCoord,
    ) -> None:
        """Complete an OD event with alighting information."""
        with self._write_lock:
            conn = self._conn()
            conn.execute(
                """
                UPDATE od_events
                SET alight_stop = ?, alight_ts = ?,
                    alight_gps_lat = ?, alight_gps_lon = ?,
                    is_complete = 1
                WHERE id = ?
                """,
                (
                    alight_stop,
                    alight_ts.isoformat(),
                    alight_gps[0],
                    alight_gps[1],
                    record_id,
                ),
            )
            # Add to sync queue
            payload = json.dumps(
                {
                    "od_event_id": record_id,
                    "alight_stop": alight_stop,
                    "alight_ts": alight_ts.isoformat(),
                }
            )
            conn.execute(
                "INSERT INTO sync_queue (od_event_id, payload) VALUES (?, ?)",
                (record_id, payload),
            )
            conn.commit()

    # ------------------------------------------------------------------
    def get_pending_sync(self, limit: int = 50) -> list[dict[str, Any]]:
        """Return OD events pending cloud sync."""
        conn = self._conn()
        rows = conn.execute(
            """
            SELECT sq.id, sq.od_event_id, sq.payload,
                   e.route_id, e.vehicle_id, e.board_stop, e.alight_stop,
                   e.board_ts, e.alight_ts
            FROM sync_queue sq
            JOIN od_events e ON e.id = sq.od_event_id
            WHERE sq.synced_at IS NULL
            ORDER BY sq.id
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]

    def mark_synced(self, sync_ids: list[int]) -> None:
        """Mark sync_queue rows as successfully uploaded."""
        if not sync_ids:
            return
        now = datetime.utcnow().isoformat()
        placeholders = ",".join("?" * len(sync_ids))
        with self._write_lock:
            conn = self._conn()
            conn.execute(
                f"UPDATE sync_queue SET synced_at = ? WHERE id IN ({placeholders})",
                [now, *sync_ids],
            )
            conn.commit()

    # ------------------------------------------------------------------
    def get_od_matrix(
        self, route_id: Optional[str] = None, vehicle_id: Optional[str] = None
    ) -> list[dict[str, Any]]:
        """
        Aggregate completed OD events into a matrix.
        Returns list of {board_stop, alight_stop, count}.
        """
        filters = ["is_complete = 1"]
        params: list[Any] = []
        if route_id:
            filters.append("route_id = ?")
            params.append(route_id)
        if vehicle_id:
            filters.append("vehicle_id = ?")
            params.append(vehicle_id)

        where = " AND ".join(filters)
        conn = self._conn()
        rows = conn.execute(
            f"""
            SELECT board_stop, alight_stop, COUNT(*) AS count
            FROM od_events
            WHERE {where}
            GROUP BY board_stop, alight_stop
            ORDER BY count DESC
            """,
            params,
        ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    def log(self, level: str, message: str) -> None:
        now = datetime.utcnow().isoformat()
        with self._write_lock:
            conn = self._conn()
            conn.execute(
                "INSERT INTO system_log (ts, level, message) VALUES (?, ?, ?)",
                (now, level, message),
            )
            conn.commit()
