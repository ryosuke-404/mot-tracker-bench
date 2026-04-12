"""
TransTrack-inspired Tracker
Sun et al., 2020 — MIT License

TransTrack's key idea: propagate detection query features from frame t
as "track queries" into frame t+1, allowing the detector itself to
maintain temporal identity. Without access to the actual TransTrack
transformer backbone (which jointly encodes detection + tracking as
attention queries), this implementation captures the algorithmic spirit:

  1. Track queries: each confirmed track is represented by its Re-ID
     embedding (analogous to TransTrack's track query feature vector).

  2. Learned association: instead of Hungarian + IoU, match track queries
     to detection queries using cosine similarity (the attention mechanism
     equivalent). No IoU gating is applied, which is TransTrack's key
     departure from SORT-family trackers.

  3. Single-stage matching: all detections at once (no high/low split),
     matching by appearance first, IoU as a tie-breaker.

  4. Query update: after matching, track query = EMA blend of matched
     detection embedding (analogous to query denoising in DINO-style
     transformer training).

  5. New queries: unmatched detections become new track queries
     (birth = detection score ≥ thresh).

Difference from original TransTrack:
  The original uses a DETR-style transformer where the detector backbone
  simultaneously produces both detection and tracking outputs. Here, we
  use external detector outputs + Re-ID embeddings to emulate the
  query-based matching without the transformer backbone weights.
"""

from __future__ import annotations

import numpy as np

from tracking.track import STrack, TrackState
from tracking.bytetrack import (
    Detection, iou_matrix, cosine_distance_matrix, linear_assignment
)

EMA_ALPHA = 0.85


class TransTracker:
    """
    TransTrack-inspired multi-object tracker.

    Drop-in replacement for ByteTracker (same update() signature).

    Matching: appearance-first (cosine), IoU as soft regularizer.
    No hard IoU gate — allows re-association after significant movement.
    """

    def __init__(
        self,
        track_thresh: float = 0.45,
        track_buffer: int = 90,
        min_hits: int = 3,
        appear_thresh: float = 0.50,    # max cosine dist for appearance match
        iou_weight: float = 0.20,       # weight of IoU in combined cost
        appear_weight: float = 0.80,    # weight of appearance in combined cost
        iou_thresh_fallback: float = 0.60,  # IoU-only fallback thresh
    ) -> None:
        self.track_thresh       = track_thresh
        self.track_buffer       = track_buffer
        self.min_hits           = min_hits
        self.appear_thresh      = appear_thresh
        self.iou_weight         = iou_weight
        self.appear_weight      = appear_weight
        self.iou_thresh_fallback = iou_thresh_fallback

        self.tracked_stracks: list[STrack] = []
        self.lost_stracks:    list[STrack] = []
        self.removed_stracks: list[STrack] = []
        self.frame_id = 0

    def reset(self) -> None:
        self.tracked_stracks.clear()
        self.lost_stracks.clear()
        self.removed_stracks.clear()
        self.frame_id = 0
        STrack.reset_id_counter()

    def update(
        self,
        high_dets: list[Detection],
        low_dets:  list[Detection],
        frame: np.ndarray | None = None,
    ) -> list[STrack]:
        self.frame_id += 1

        all_dets = high_dets + low_dets
        all_tracks = self.tracked_stracks + self.lost_stracks
        STrack.multi_predict(all_tracks)

        # -- Step 1: appearance-first matching (track query × det query) --
        confirmed = [t for t in all_tracks if t.state == TrackState.Confirmed]
        tentative = [t for t in self.tracked_stracks if t.state == TrackState.Tentative]

        (m1, unm_c, unm_d1) = self._associate_query(
            high_dets, confirmed, self.appear_thresh
        )
        matched_c = {t for t, _ in m1}

        for t_idx, d_idx in m1:
            track = confirmed[t_idx]
            det   = high_dets[d_idx]
            track.update(det.bbox, det.score, None)
            if det.embedding is not None:
                self._ema_update(track, det.embedding)

        # -- Step 2: IoU fallback for unmatched confirmed + all tentative --
        fallback_tracks = [confirmed[i] for i in unm_c] + tentative
        remaining_high  = [high_dets[i] for i in unm_d1]

        (m2, unm_f, unm_d2) = self._associate_iou(
            remaining_high, fallback_tracks, self.iou_thresh_fallback
        )
        for t_idx, d_idx in m2:
            track = fallback_tracks[t_idx]
            det   = remaining_high[d_idx]
            track.update(det.bbox, det.score, None)
            if det.embedding is not None:
                self._ema_update(track, det.embedding)

        unmatched_active = [fallback_tracks[i] for i in unm_f]

        # -- Step 3: low dets vs unmatched + lost (IoU only) --------------
        candidate_s3 = unmatched_active + self.lost_stracks
        (m3, unm_s3, _) = self._associate_iou(
            low_dets, candidate_s3, 1.0 - 0.45
        )
        matched_s3 = {t for t, _ in m3}
        for t_idx, d_idx in m3:
            candidate_s3[t_idx].update(low_dets[d_idx].bbox, low_dets[d_idx].score, None)
        for i, track in enumerate(candidate_s3):
            if i not in matched_s3:
                if track.state != TrackState.Lost:
                    track.mark_lost()
                else:
                    track.increment_age()

        # -- Step 4: new query tracks for unmatched high dets -------------
        already_high_remaining = {unm_d1[j] for j in unm_d2}
        for orig_idx in already_high_remaining:
            det = high_dets[orig_idx]
            if det.score < self.track_thresh:
                continue
            t = STrack(det.bbox, det.score)
            t.activate(self.frame_id)
            if det.embedding is not None:
                t.reid_embedding = det.embedding.copy()
                t.embedding_history.append(det.embedding)
            self.tracked_stracks.append(t)

        # -- Rebuild -------------------------------------------------------
        seen: set[int] = set()
        still_lost: list[STrack] = []
        for track in (self.lost_stracks
                      + [t for t in unmatched_active if t.state == TrackState.Lost]):
            if track.time_since_update > self.track_buffer:
                track.mark_removed()
                self.removed_stracks.append(track)
            elif track.track_id not in seen:
                still_lost.append(track)
                seen.add(track.track_id)
        self.lost_stracks = still_lost
        self.tracked_stracks = [
            t for t in self.tracked_stracks
            if t.state in (TrackState.Confirmed, TrackState.Tentative)
        ]
        return list(self.tracked_stracks)

    def get_confirmed_tracks(self) -> list[STrack]:
        return [t for t in self.tracked_stracks if t.state == TrackState.Confirmed]

    def get_lost_tracks(self) -> list[STrack]:
        return list(self.lost_stracks)

    # ------------------------------------------------------------------

    def _associate_query(
        self,
        detections: list[Detection],
        tracks: list[STrack],
        thresh: float,
    ) -> tuple:
        """
        Appearance-first matching: cosine distance + soft IoU regularizer.
        Tracks or dets without embeddings fall back to IoU-only.
        """
        if not tracks or not detections:
            return [], list(range(len(tracks))), list(range(len(detections)))

        track_boxes = np.array([t.tlbr for t in tracks])
        det_boxes   = np.array([d.bbox for d in detections])
        iou_d = 1.0 - iou_matrix(track_boxes, det_boxes)

        t_embs = [t.reid_embedding for t in tracks]
        d_embs = [d.embedding      for d in detections]
        if all(e is not None for e in t_embs) and all(e is not None for e in d_embs):
            reid_d = cosine_distance_matrix(np.stack(t_embs), np.stack(d_embs))
            cost   = self.appear_weight * reid_d + self.iou_weight * iou_d
        else:
            cost = iou_d

        return linear_assignment(cost, thresh)

    def _associate_iou(
        self,
        detections: list[Detection],
        tracks: list[STrack],
        thresh: float,
    ) -> tuple:
        if not tracks or not detections:
            return [], list(range(len(tracks))), list(range(len(detections)))
        cost = 1.0 - iou_matrix(
            np.array([t.tlbr for t in tracks]),
            np.array([d.bbox for d in detections]),
        )
        return linear_assignment(cost, thresh)

    def _ema_update(self, track: STrack, embed: np.ndarray) -> None:
        if track.reid_embedding is None:
            track.reid_embedding = embed.copy()
        else:
            track.reid_embedding = (EMA_ALPHA * track.reid_embedding
                                    + (1 - EMA_ALPHA) * embed)
            n = np.linalg.norm(track.reid_embedding)
            if n > 1e-12:
                track.reid_embedding /= n
        track.embedding_history.append(track.reid_embedding.copy())
