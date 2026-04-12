"""
DeepSORT: Simple Online and Realtime Tracking with a Deep Association Metric
Wojke et al., ICIP 2017 — MIT License

Key innovations over SORT:
  1. Deep appearance descriptor (Re-ID embedding) as matching metric.
  2. Cascaded matching: confirmed tracks with smallest age matched first,
     reducing the chance of a newly-created track stealing an established ID.
  3. Mahalanobis gating: reject (track, detection) pairs where the Kalman
     prediction is too far from the detection in state space.
  4. Combined cost: gate with Mahalanobis, rank by cosine distance.
  5. Appearance gallery: each track stores a deque of past embeddings;
     minimum cosine distance over the gallery is used as appearance cost.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from tracking.track import STrack, TrackState
from tracking.bytetrack import Detection, iou_matrix, cosine_distance_matrix, linear_assignment
from tracking.kalman_filter import KalmanFilter

_shared_kalman = KalmanFilter()

# Mahalanobis gating threshold (chi-sq 95%, DOF=4)
_MAHA_GATE = KalmanFilter.chi2inv95[4]   # 9.4877

# Maximum cosine distance for a match to be accepted
_MAX_COSINE_DIST = 0.25


# ---------------------------------------------------------------------------
# Per-track appearance gallery
# ---------------------------------------------------------------------------

class AppearanceGallery:
    """Rolling gallery of Re-ID embeddings for one track."""

    def __init__(self, max_size: int = 100) -> None:
        self._feats: deque[np.ndarray] = deque(maxlen=max_size)

    def add(self, feat: np.ndarray) -> None:
        self._feats.append(feat)

    def min_cosine_dist(self, query: np.ndarray) -> float:
        """Minimum cosine distance between query and any stored embedding."""
        if not self._feats:
            return 1.0
        gallery = np.stack(list(self._feats))         # (K, D)
        sim = gallery @ query                          # (K,)  (both L2-normed)
        return float(1.0 - np.max(sim))

    def __len__(self) -> int:
        return len(self._feats)


# ---------------------------------------------------------------------------
# DeepSORT Tracker
# ---------------------------------------------------------------------------

class DeepSORTTracker:
    """
    DeepSORT multi-object tracker.

    Drop-in replacement for ByteTracker (same update() signature).

    Matching pipeline:
      1. Cascade match: confirmed tracks (age 1 → max_age) × high dets
         Cost = cosine distance, gated by Mahalanobis distance
      2. IoU match: remaining confirmed + all tentative tracks × remaining high dets
      3. IoU match: unmatched tracks from (1+2) × low dets
      4. New tentative tracks for still-unmatched high dets
    """

    def __init__(
        self,
        track_thresh: float = 0.45,
        track_buffer: int = 90,
        min_hits: int = 3,
        max_cosine_dist: float = _MAX_COSINE_DIST,
        nn_budget: int = 100,           # max embeddings per track gallery
        iou_thresh_fallback: float = 0.70,
        iou_thresh_stage2: float = 0.45,
    ) -> None:
        self.track_thresh        = track_thresh
        self.track_buffer        = track_buffer
        self.min_hits            = min_hits
        self.max_cosine_dist     = max_cosine_dist
        self.nn_budget           = nn_budget
        self.iou_thresh_fallback = iou_thresh_fallback
        self.iou_thresh_stage2   = iou_thresh_stage2

        self.tracked_stracks: list[STrack] = []
        self.lost_stracks:    list[STrack] = []
        self.removed_stracks: list[STrack] = []
        # Per-track appearance gallery
        self._galleries: dict[int, AppearanceGallery] = {}
        self.frame_id = 0

    # ------------------------------------------------------------------
    def reset(self) -> None:
        self.tracked_stracks.clear()
        self.lost_stracks.clear()
        self.removed_stracks.clear()
        self._galleries.clear()
        self.frame_id = 0
        STrack.reset_id_counter()

    # ------------------------------------------------------------------
    def update(
        self,
        high_dets: list[Detection],
        low_dets:  list[Detection],
        frame: np.ndarray | None = None,
    ) -> list[STrack]:
        self.frame_id += 1

        # Predict Kalman for all tracks
        all_tracks = self.tracked_stracks + self.lost_stracks
        STrack.multi_predict(all_tracks)

        # Separate confirmed vs tentative active tracks
        confirmed  = [t for t in self.tracked_stracks if t.state == TrackState.Confirmed]
        tentative  = [t for t in self.tracked_stracks if t.state == TrackState.Tentative]

        # ---- Step 1: Cascade matching (confirmed tracks × high dets) -----
        (matches_cascade, unmatched_confirmed_idx,
         unmatched_high_after_cascade) = self._cascade_match(confirmed, high_dets)

        for t_idx, d_idx in matches_cascade:
            t = confirmed[t_idx]
            d = high_dets[d_idx]
            t.update(d.bbox, d.score, d.embedding)
            self._update_gallery(t, d.embedding)

        # ---- Step 2: IoU fallback for unmatched confirmed + tentative ----
        remaining_tracks = [confirmed[i] for i in unmatched_confirmed_idx] + tentative
        remaining_high   = [high_dets[i] for i in unmatched_high_after_cascade]

        (matches_iou, unmatched_rt_idx,
         unmatched_high_idx) = self._iou_match(remaining_tracks, remaining_high,
                                                self.iou_thresh_fallback)

        for t_idx, d_idx in matches_iou:
            t = remaining_tracks[t_idx]
            d = remaining_high[d_idx]
            t.update(d.bbox, d.score, d.embedding)
            self._update_gallery(t, d.embedding)

        unmatched_active = [remaining_tracks[i] for i in unmatched_rt_idx]

        # ---- Step 3: Low-score dets vs unmatched active + lost (IoU) ----
        candidate_s3 = unmatched_active + self.lost_stracks
        (matches_s3, unmatched_s3_idx, _) = self._iou_match(
            candidate_s3, low_dets, self.iou_thresh_stage2
        )
        matched_s3 = {t for t, _ in matches_s3}
        for t_idx, d_idx in matches_s3:
            candidate_s3[t_idx].update(low_dets[d_idx].bbox,
                                        low_dets[d_idx].score, None)

        for i, track in enumerate(candidate_s3):
            if i not in matched_s3:
                if track.state != TrackState.Lost:
                    track.mark_lost()
                else:
                    track.increment_age()

        # ---- Step 4: New tentative tracks --------------------------------
        for d_idx in unmatched_high_idx:
            det = high_dets[d_idx] if unmatched_high_after_cascade else remaining_high[d_idx]
            # re-index back into high_dets
            actual_det = high_dets[unmatched_high_after_cascade[d_idx]] \
                if unmatched_high_after_cascade else remaining_high[d_idx]
            if actual_det.score < self.track_thresh:
                continue
            new_track = STrack(actual_det.bbox, actual_det.score)
            new_track.activate(self.frame_id)
            if actual_det.embedding is not None:
                new_track.reid_embedding = actual_det.embedding
                new_track.embedding_history.append(actual_det.embedding)
                self._update_gallery(new_track, actual_det.embedding)
            self.tracked_stracks.append(new_track)

        # ---- Rebuild lists -----------------------------------------------
        still_lost: list[STrack] = []
        for track in (self.lost_stracks
                      + [t for t in unmatched_active if t.state == TrackState.Lost]):
            if track.time_since_update > self.track_buffer:
                track.mark_removed()
                self.removed_stracks.append(track)
                self._galleries.pop(track.track_id, None)
            elif track.track_id not in {t.track_id for t in still_lost}:
                still_lost.append(track)

        self.lost_stracks = still_lost
        self.tracked_stracks = [
            t for t in self.tracked_stracks
            if t.state in (TrackState.Confirmed, TrackState.Tentative)
        ]
        return list(self.tracked_stracks)

    # ------------------------------------------------------------------
    def get_confirmed_tracks(self) -> list[STrack]:
        return [t for t in self.tracked_stracks if t.state == TrackState.Confirmed]

    def get_lost_tracks(self) -> list[STrack]:
        return list(self.lost_stracks)

    # ------------------------------------------------------------------
    # Cascade matching
    # ------------------------------------------------------------------

    def _cascade_match(
        self,
        confirmed: list[STrack],
        detections: list[Detection],
    ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        """
        Match confirmed tracks to detections in age order (1, 2, ..., max_age).
        Uses cosine distance gated by Mahalanobis distance.
        """
        if not confirmed or not detections:
            return [], list(range(len(confirmed))), list(range(len(detections)))

        unmatched_det_idx = list(range(len(detections)))
        unmatched_trk_idx = list(range(len(confirmed)))
        all_matches: list[tuple[int, int]] = []

        det_embeds = [d.embedding for d in detections]
        has_embeds = all(e is not None for e in det_embeds)

        # Sort confirmed tracks by age (smallest time_since_update first)
        age_groups: dict[int, list[int]] = {}
        for i, t in enumerate(confirmed):
            age = t.time_since_update
            age_groups.setdefault(age, []).append(i)

        for age in sorted(age_groups):
            t_indices = [i for i in age_groups[age] if i in set(unmatched_trk_idx)]
            if not t_indices or not unmatched_det_idx:
                break

            d_indices = list(unmatched_det_idx)
            cost = self._appearance_cost(
                [confirmed[i] for i in t_indices],
                [detections[j] for j in d_indices],
                has_embeds,
            )
            # Gate with Mahalanobis distance
            self._apply_maha_gate(
                cost,
                [confirmed[i] for i in t_indices],
                [detections[j] for j in d_indices],
            )

            matches_raw, unm_t_raw, unm_d_raw = linear_assignment(
                cost, self.max_cosine_dist
            )
            for local_t, local_d in matches_raw:
                global_t = t_indices[local_t]
                global_d = d_indices[local_d]
                all_matches.append((global_t, global_d))
                unmatched_trk_idx.remove(global_t)
                unmatched_det_idx.remove(global_d)

        return all_matches, unmatched_trk_idx, unmatched_det_idx

    def _appearance_cost(
        self,
        tracks: list[STrack],
        detections: list[Detection],
        has_embeds: bool,
    ) -> np.ndarray:
        """Cost matrix based on appearance (cosine min distance per gallery)."""
        M, N = len(tracks), len(detections)
        cost = np.ones((M, N), dtype=np.float64)
        if not has_embeds:
            return cost
        for i, track in enumerate(tracks):
            gallery = self._galleries.get(track.track_id)
            if gallery is None or len(gallery) == 0:
                continue
            for j, det in enumerate(detections):
                if det.embedding is not None:
                    cost[i, j] = gallery.min_cosine_dist(det.embedding)
        return cost

    def _apply_maha_gate(
        self,
        cost: np.ndarray,
        tracks: list[STrack],
        detections: list[Detection],
    ) -> None:
        """Set cost = INF for pairs exceeding Mahalanobis gating threshold."""
        det_measurements = np.array([STrack.tlbr_to_xyah(d.bbox) for d in detections])
        for i, track in enumerate(tracks):
            if track.mean is None:
                continue
            dists = _shared_kalman.gating_distance(
                track.mean, track.covariance, det_measurements
            )
            cost[i, dists > _MAHA_GATE] = 1.0   # gate out

    def _iou_match(
        self,
        tracks: list[STrack],
        detections: list[Detection],
        thresh: float,
    ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        if not tracks or not detections:
            return [], list(range(len(tracks))), list(range(len(detections)))
        track_boxes = np.array([t.tlbr for t in tracks])
        det_boxes   = np.array([d.bbox for d in detections])
        cost = 1.0 - iou_matrix(track_boxes, det_boxes)
        return linear_assignment(cost, thresh)

    # ------------------------------------------------------------------

    def _update_gallery(self, track: STrack, embedding: np.ndarray | None) -> None:
        if embedding is None:
            return
        if track.track_id not in self._galleries:
            self._galleries[track.track_id] = AppearanceGallery(self.nn_budget)
        self._galleries[track.track_id].add(embedding)
