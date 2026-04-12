"""
FrameProcessor: main pipeline orchestrator.

Per-frame processing order:
  1. RT-DETRv2 detection → high_dets, low_dets
  2. FastReID feature extraction for high-score detections
  3. ByteTrack update (2-stage matching + Kalman)
  4. Re-ID gallery update for confirmed tracks
  5. Re-ID rescue: re-identify new tentative tracks against lost gallery entries
  6. Normalize centroids and run tripwire crossing checks
  7. Dispatch crossing events → ODTracker
  8. Periodic gallery pruning
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np

from detection.rtdetr_detector import RTDETRv2Detector, RawDetection
from tracking.bytetrack import ByteTracker, Detection as TrackDetection
from tracking.track import STrack, TrackState
from reid.fastreid_extractor import FastReIDExtractor
from reid.gallery import ReIDGallery
from tripwire.tripwire_manager import TripwireManager, CrossingEvent
from od.od_tracker import ODTracker

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class FrameResult:
    frame_id: int
    timestamp: datetime
    active_tracks: list[STrack]
    crossing_events: list[CrossingEvent]
    completed_od_records: list        # list[PassengerRecord]
    inference_ms: float = 0.0
    tracking_ms: float = 0.0
    reid_ms: float = 0.0


# ---------------------------------------------------------------------------
# FrameProcessor
# ---------------------------------------------------------------------------

class FrameProcessor:
    """
    Stateful pipeline that processes one frame at a time.

    Instantiate once and call `process_frame()` in a loop.
    """

    def __init__(
        self,
        camera_id: str,
        detector: RTDETRv2Detector,
        tracker: ByteTracker,
        reid_extractor: FastReIDExtractor,
        reid_gallery: ReIDGallery,
        tripwire_manager: TripwireManager,
        od_tracker: ODTracker,
        gps_reader,               # GpsReader | MockGpsReader
        frame_width: int,
        frame_height: int,
        gallery_prune_interval: int = 100,
        gallery_max_age_frames: int = 3000,
    ) -> None:
        self.camera_id = camera_id
        self.detector = detector
        self.tracker = tracker
        self.reid_extractor = reid_extractor
        self.reid_gallery = reid_gallery
        self.tripwire_manager = tripwire_manager
        self.od_tracker = od_tracker
        self.gps_reader = gps_reader
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.gallery_prune_interval = gallery_prune_interval
        self.gallery_max_age_frames = gallery_max_age_frames

        self._frame_id = 0
        # track_id → centroid in NORMALIZED coordinates from previous frame
        self._prev_centroids: dict[int, tuple[float, float]] = {}

    # ------------------------------------------------------------------
    def process_frame(
        self, frame: np.ndarray, timestamp: Optional[datetime] = None
    ) -> FrameResult:
        """
        Process a single BGR frame.

        Args:
            frame      BGR image from OpenCV
            timestamp  datetime for this frame; uses utcnow() if None
        """
        import time

        if timestamp is None:
            timestamp = datetime.utcnow()

        self._frame_id += 1
        fid = self._frame_id

        # ---- Step 1: Detection -----------------------------------------
        t0 = time.perf_counter()
        high_raw, low_raw = self.detector.detect(frame)
        t1 = time.perf_counter()
        inference_ms = (t1 - t0) * 1000

        # ---- Step 2: Re-ID feature extraction for high-score dets ------
        t2 = time.perf_counter()
        high_embeddings = self._extract_embeddings(frame, high_raw)
        reid_ms = (time.perf_counter() - t2) * 1000

        # Convert to ByteTrack Detection objects
        high_dets = [
            TrackDetection(bbox=d.bbox, score=d.score, embedding=emb)
            for d, emb in zip(high_raw, high_embeddings)
        ]
        low_dets = [
            TrackDetection(bbox=d.bbox, score=d.score)
            for d in low_raw
        ]

        # ---- Step 3: ByteTrack update -----------------------------------
        t3 = time.perf_counter()
        active_tracks = self.tracker.update(high_dets, low_dets)
        tracking_ms = (time.perf_counter() - t3) * 1000

        # ---- Step 4: Update Re-ID gallery for confirmed tracks ----------
        confirmed_tracks = [t for t in active_tracks if t.state == TrackState.Confirmed]
        for track in confirmed_tracks:
            if track.reid_embedding is not None:
                self.reid_gallery.add_embedding(
                    track.track_id, track.reid_embedding, fid
                )

        # ---- Step 5: Re-ID rescue for tentative tracks ------------------
        self._reid_rescue(active_tracks, frame, high_dets, fid)

        # ---- Step 6: Tripwire crossing detection ------------------------
        curr_centroids = {
            t.track_id: self._normalize_centroid(t.centroid)
            for t in active_tracks
        }
        crossing_events = self.tripwire_manager.update(
            camera_id=self.camera_id,
            frame_id=fid,
            track_positions=curr_centroids,
            prev_positions=self._prev_centroids,
        )
        self._prev_centroids = curr_centroids

        # ---- Step 7: Dispatch crossing events to OD tracker -------------
        current_gps = self.gps_reader.latest_coord
        completed_records = []
        for event in crossing_events:
            result = self.od_tracker.process_event(event, timestamp, current_gps)
            if result is not None:
                completed_records.append(result)

        # ---- Step 8: Periodic gallery pruning ---------------------------
        if fid % self.gallery_prune_interval == 0:
            active_ids = {t.track_id for t in active_tracks}
            lost_ids = {t.track_id for t in self.tracker.get_lost_tracks()}
            keep_ids = active_ids | lost_ids
            pruned = self.reid_gallery.prune_old_tracks(
                active_ids=keep_ids,
                max_age_frames=self.gallery_max_age_frames,
                current_frame=fid,
            )
            if pruned:
                logger.debug("Frame %d: pruned %d gallery entries", fid, pruned)

        return FrameResult(
            frame_id=fid,
            timestamp=timestamp,
            active_tracks=active_tracks,
            crossing_events=crossing_events,
            completed_od_records=completed_records,
            inference_ms=inference_ms,
            tracking_ms=tracking_ms,
            reid_ms=reid_ms,
        )

    # ------------------------------------------------------------------
    # Re-ID rescue
    # ------------------------------------------------------------------

    def _reid_rescue(
        self,
        active_tracks: list[STrack],
        frame: np.ndarray,
        high_dets: list[TrackDetection],
        frame_id: int,
    ) -> None:
        """
        For newly created tentative tracks (hits == 1), check whether the
        gallery can match them to a previously lost track.

        If a match is found above the gallery threshold:
          - Reassign the new STrack to the old track_id
          - Notify od_tracker so boarding records are transferred
        """
        new_tentative = [
            t for t in active_tracks
            if t.state == TrackState.Tentative and t.hits == 1
        ]
        if not new_tentative:
            return

        active_ids = {t.track_id for t in active_tracks}

        for track in new_tentative:
            if track.reid_embedding is None:
                continue

            matched_id, sim = self.reid_gallery.query(
                track.reid_embedding, exclude_ids=active_ids
            )
            if matched_id is None:
                continue

            old_id = matched_id
            new_id = track.track_id

            logger.info(
                "Re-ID rescue: new track %d → restored to id %d (sim=%.3f, frame=%d)",
                new_id, old_id, sim, frame_id
            )

            # Restore the original track_id on the STrack object so the
            # displayed ID in the video stays consistent for the same person.
            track.reassign_id(old_id)

            # OD record was under old_id — it continues under old_id (no transfer needed)
            # (resurrect_track is only needed when the STrack keeps the new_id)

            # Merge gallery: keep embeddings under old_id, discard the new_id slot
            self.reid_gallery.remove_track(new_id)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _extract_embeddings(
        self, frame: np.ndarray, detections: list[RawDetection]
    ) -> list[np.ndarray | None]:
        if not detections:
            return []
        bboxes = [d.bbox for d in detections]
        embeddings = self.reid_extractor.extract_batch(frame, bboxes)
        return [embeddings[i] for i in range(len(detections))]

    def _normalize_centroid(
        self, centroid: tuple[float, float]
    ) -> tuple[float, float]:
        """Convert pixel centroid to normalized [0, 1] coordinates."""
        cx, cy = centroid
        return cx / self.frame_width, cy / self.frame_height
