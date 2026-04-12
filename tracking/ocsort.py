"""
OC-SORT: Observation-Centric SORT
Cao et al., CVPR 2023 — MIT License

Key innovations over SORT:
  1. OCM (Observation-Centric Momentum)
     Adds a velocity-consistency penalty to the cost matrix.
     Tracks that have been moving consistently receive lower cost
     when matched with detections that continue that motion.

  2. OCR (Observation-Centric Re-update)
     When a lost track is re-associated, the Kalman state is
     re-initialized from the last real observation rather than
     from the accumulated (drifted) Kalman prediction.
     This corrects state drift during occlusion.

  3. Virtual trajectory smoothing
     During unobserved frames, track position is interpolated
     linearly between last observation and current prediction,
     making the IoU matching more robust.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from tracking.track import STrack, TrackState
from tracking.bytetrack import Detection, iou_matrix, cosine_distance_matrix, linear_assignment
from tracking.kalman_filter import KalmanFilter


# ---------------------------------------------------------------------------
# OC-SORT per-track state extension
# ---------------------------------------------------------------------------

class OCSTrack(STrack):
    """
    Extends STrack with observation history for OCM/OCR.
    Stores the last two observed bounding boxes (not Kalman predictions).
    """

    def __init__(self, tlbr: np.ndarray, score: float) -> None:
        super().__init__(tlbr, score)
        # Ring buffer of raw observations: each entry is tlbr
        self.observations: deque[np.ndarray] = deque(maxlen=2)
        self.observations.append(tlbr.copy())
        # Smoothed velocity estimate (cx, cy) from last two observations
        self._obs_velocity: np.ndarray = np.zeros(2, dtype=np.float64)

    def update_obs_velocity(self) -> None:
        """Recompute observation-centric velocity from last two observations."""
        if len(self.observations) == 2:
            b_prev, b_curr = self.observations[0], self.observations[1]
            cx_prev = (b_prev[0] + b_prev[2]) / 2
            cy_prev = (b_prev[1] + b_prev[3]) / 2
            cx_curr = (b_curr[0] + b_curr[2]) / 2
            cy_curr = (b_curr[1] + b_curr[3]) / 2
            self._obs_velocity = np.array([cx_curr - cx_prev,
                                           cy_curr - cy_prev], dtype=np.float64)

    def obs_velocity(self) -> np.ndarray:
        return self._obs_velocity

    def last_observation(self) -> np.ndarray:
        return self.observations[-1]

    def add_observation(self, tlbr: np.ndarray) -> None:
        self.observations.append(tlbr.copy())
        self.update_obs_velocity()


# ---------------------------------------------------------------------------
# OC-SORT Tracker
# ---------------------------------------------------------------------------

class OCSORTTracker:
    """
    OC-SORT multi-object tracker.

    Drop-in replacement for ByteTracker.
    Two-stage matching (high/low) like ByteTrack, plus OCM cost correction
    and OCR state re-initialization for re-associated lost tracks.
    """

    def __init__(
        self,
        track_thresh: float = 0.45,
        track_buffer: int = 90,
        match_thresh: float = 0.85,
        min_hits: int = 3,
        iou_thresh_stage2: float = 0.45,
        reid_cost_weight: float = 0.35,
        ocm_weight: float = 0.20,     # velocity consistency penalty weight
        delta_t: int = 3,             # frame window for velocity estimation
    ) -> None:
        self.track_thresh      = track_thresh
        self.track_buffer      = track_buffer
        self.match_thresh      = match_thresh
        self.min_hits          = min_hits
        self.iou_thresh_stage2 = iou_thresh_stage2
        self.reid_cost_weight  = reid_cost_weight
        self.ocm_weight        = ocm_weight
        self.delta_t           = delta_t

        self.tracked_stracks: list[OCSTrack] = []
        self.lost_stracks:    list[OCSTrack] = []
        self.removed_stracks: list[OCSTrack] = []
        self.frame_id = 0

    # ------------------------------------------------------------------
    def reset(self) -> None:
        self.tracked_stracks.clear()
        self.lost_stracks.clear()
        self.removed_stracks.clear()
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
        all_active = self.tracked_stracks + self.lost_stracks
        STrack.multi_predict(all_active)

        # ---- Stage 1: high dets vs active tracks (OCM + IoU + ReID) -----
        (matches_s1, unmatched_t_s1,
         unmatched_high_idx) = self._associate_ocm(
            high_dets, self.tracked_stracks, self.match_thresh
        )

        for t_idx, d_idx in matches_s1:
            track = self.tracked_stracks[t_idx]
            det   = high_dets[d_idx]
            # OCR: if track was lost and re-found, re-init Kalman from last obs
            if track.time_since_update > 0:
                self._ocr_reinit(track, det.bbox)
            track.update(det.bbox, det.score, det.embedding)
            track.add_observation(det.bbox)

        unmatched_active = [self.tracked_stracks[i] for i in unmatched_t_s1]

        # ---- Stage 2: low dets vs unmatched active + lost (IoU only) ----
        candidate_s2 = unmatched_active + self.lost_stracks
        (matches_s2, unmatched_s2_idx,
         _) = self._associate_iou_only(low_dets, candidate_s2, self.iou_thresh_stage2)

        matched_s2 = {t_idx for t_idx, _ in matches_s2}
        for t_idx, d_idx in matches_s2:
            track = candidate_s2[t_idx]
            det   = low_dets[d_idx]
            if track.time_since_update > 0:
                self._ocr_reinit(track, det.bbox)
            track.update(det.bbox, det.score, None)
            track.add_observation(det.bbox)

        for i, track in enumerate(candidate_s2):
            if i not in matched_s2:
                if track.state != TrackState.Lost:
                    track.mark_lost()
                else:
                    track.increment_age()

        # ---- New tentative tracks ----------------------------------------
        for d_idx in unmatched_high_idx:
            det = high_dets[d_idx]
            if det.score < self.track_thresh:
                continue
            new_track = OCSTrack(det.bbox, det.score)
            new_track.activate(self.frame_id)
            if det.embedding is not None:
                new_track.reid_embedding = det.embedding
                new_track.embedding_history.append(det.embedding)
            self.tracked_stracks.append(new_track)

        # ---- Rebuild lists; remove expired lost tracks -------------------
        still_lost: list[OCSTrack] = []
        for track in self.lost_stracks + [
            t for t in unmatched_active if t.state == TrackState.Lost
        ]:
            if track.time_since_update > self.track_buffer:
                track.mark_removed()
                self.removed_stracks.append(track)
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
    # OCR: re-initialize Kalman from last real observation
    # ------------------------------------------------------------------

    def _ocr_reinit(self, track: OCSTrack, new_bbox: np.ndarray) -> None:
        """
        When a lost track is re-found, reset its Kalman state by linearly
        interpolating from its last observation to the new detection.
        This corrects drift accumulated during the occlusion period.
        """
        if not track.observations:
            return
        last_obs = track.last_observation()
        # Re-initialize mean from last observation (not drifted Kalman state)
        from tracking.kalman_filter import KalmanFilter as KF
        kf = KF()
        last_xyah = STrack.tlbr_to_xyah(last_obs)
        new_xyah  = STrack.tlbr_to_xyah(new_bbox)
        # Velocity from last obs → current detection
        vel = new_xyah - last_xyah
        mean = np.r_[last_xyah, vel]
        mean, cov = kf.initiate(last_xyah)
        # Inject velocity estimate into state
        mean[4:8] = vel
        track.mean = mean
        track.covariance = cov

    # ------------------------------------------------------------------
    # OCM cost: IoU + velocity consistency penalty + optional ReID
    # ------------------------------------------------------------------

    def _associate_ocm(
        self,
        detections: list[Detection],
        tracks: list[OCSTrack],
        thresh: float,
    ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        if not tracks or not detections:
            return [], list(range(len(tracks))), list(range(len(detections)))

        track_boxes = np.array([t.tlbr for t in tracks])
        det_boxes   = np.array([d.bbox for d in detections])

        iou   = iou_matrix(track_boxes, det_boxes)   # (M, N)
        iou_d = 1.0 - iou

        # OCM: velocity consistency penalty
        ocm_penalty = self._ocm_penalty(tracks, det_boxes)  # (M, N) in [0,1]

        # Optional ReID
        t_embeds = [t.reid_embedding for t in tracks]
        d_embeds = [d.embedding      for d in detections]
        if all(e is not None for e in t_embeds) and all(e is not None for e in d_embeds):
            T = np.stack(t_embeds)
            D = np.stack(d_embeds)
            reid_d = cosine_distance_matrix(T, D)
            cost = (
                (1.0 - self.reid_cost_weight - self.ocm_weight) * iou_d
                + self.reid_cost_weight * reid_d
                + self.ocm_weight * ocm_penalty
            )
        else:
            cost = (1.0 - self.ocm_weight) * iou_d + self.ocm_weight * ocm_penalty

        return linear_assignment(cost, thresh)

    def _ocm_penalty(
        self, tracks: list[OCSTrack], det_boxes: np.ndarray
    ) -> np.ndarray:
        """
        Observation-Centric Momentum penalty matrix (M, N).

        For each (track, detection) pair:
          - Compute detection velocity relative to track's last observation
          - Compare with track's historical velocity
          - Penalty = normalized velocity mismatch in [0, 1]
        """
        M = len(tracks)
        N = det_boxes.shape[0]
        penalty = np.zeros((M, N), dtype=np.float64)

        for i, track in enumerate(tracks):
            if len(track.observations) < 1:
                continue
            last_obs = track.last_observation()
            last_cx = (last_obs[0] + last_obs[2]) / 2
            last_cy = (last_obs[1] + last_obs[3]) / 2
            v_track = track.obs_velocity()  # (2,) historical velocity

            for j in range(N):
                det_cx = (det_boxes[j, 0] + det_boxes[j, 2]) / 2
                det_cy = (det_boxes[j, 1] + det_boxes[j, 3]) / 2
                v_det  = np.array([det_cx - last_cx, det_cy - last_cy])

                diff   = np.linalg.norm(v_det - v_track)
                scale  = max(np.linalg.norm(v_track), np.linalg.norm(v_det), 1.0)
                penalty[i, j] = min(diff / scale, 1.0)

        return penalty

    def _associate_iou_only(
        self,
        detections: list[Detection],
        tracks: list[OCSTrack],
        thresh: float,
    ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        if not tracks or not detections:
            return [], list(range(len(tracks))), list(range(len(detections)))
        track_boxes = np.array([t.tlbr for t in tracks])
        det_boxes   = np.array([d.bbox for d in detections])
        cost = 1.0 - iou_matrix(track_boxes, det_boxes)
        return linear_assignment(cost, thresh)
