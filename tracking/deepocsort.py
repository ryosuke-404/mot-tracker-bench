"""
Deep OC-SORT: Multi-Object Tracking with Observation-Centric SORT
  Enhanced with Deep Appearance Descriptors
Maggiolino et al., 2023 — MIT License

Combines OC-SORT's geometric improvements with DeepSORT's appearance:
  1. OCM (Observation-Centric Momentum): velocity consistency penalty
  2. OCR (Observation-Centric Re-update): re-init Kalman on re-match
  3. Deep appearance cost (cosine) with EMA gallery embeddings
  4. Two-stage matching (high/low) like ByteTrack

Cost matrix (stage 1):
  C = (1 - α - β) * IoU_dist  +  α * Re-ID_dist  +  β * OCM_penalty

Stage 2: IoU only (same as ByteTrack).
"""

from __future__ import annotations

from collections import deque

import numpy as np

from tracking.track import STrack, TrackState
from tracking.bytetrack import (
    Detection, iou_matrix, cosine_distance_matrix, linear_assignment
)
from tracking.kalman_filter import KalmanFilter

_kf = KalmanFilter()

EMA_ALPHA = 0.9


# ---------------------------------------------------------------------------

class DOCSTrack(STrack):
    """STrack extended with observation history and EMA embedding."""

    def __init__(self, tlbr: np.ndarray, score: float) -> None:
        super().__init__(tlbr, score)
        self.obs_history: deque[np.ndarray] = deque(maxlen=2)
        self.obs_history.append(tlbr.copy())
        self._obs_vel: np.ndarray = np.zeros(2, dtype=np.float64)
        self._ema_embed: np.ndarray | None = None

    # -- observation velocity ---------------------------------------------
    def push_obs(self, tlbr: np.ndarray) -> None:
        self.obs_history.append(tlbr.copy())
        if len(self.obs_history) == 2:
            b0, b1 = self.obs_history[0], self.obs_history[1]
            self._obs_vel = np.array([
                (b1[0]+b1[2])/2 - (b0[0]+b0[2])/2,
                (b1[1]+b1[3])/2 - (b0[1]+b0[3])/2,
            ])

    def obs_velocity(self) -> np.ndarray:
        return self._obs_vel

    def last_obs(self) -> np.ndarray:
        return self.obs_history[-1]

    # -- EMA appearance ---------------------------------------------------
    def update_ema(self, new_embed: np.ndarray) -> None:
        if self._ema_embed is None:
            self._ema_embed = new_embed.copy()
        else:
            self._ema_embed = EMA_ALPHA * self._ema_embed + (1 - EMA_ALPHA) * new_embed
            n = np.linalg.norm(self._ema_embed)
            if n > 1e-12:
                self._ema_embed /= n
        self.reid_embedding = self._ema_embed
        self.embedding_history.append(self._ema_embed.copy())


# ---------------------------------------------------------------------------

class DeepOCSORTTracker:
    """
    Deep OC-SORT multi-object tracker.

    Drop-in replacement for ByteTracker (same update() signature).
    """

    def __init__(
        self,
        track_thresh: float = 0.45,
        track_buffer: int = 90,
        match_thresh: float = 0.85,
        min_hits: int = 3,
        iou_thresh_stage2: float = 0.45,
        reid_weight: float = 0.40,
        ocm_weight:  float = 0.15,
    ) -> None:
        self.track_thresh      = track_thresh
        self.track_buffer      = track_buffer
        self.match_thresh      = match_thresh
        self.min_hits          = min_hits
        self.iou_thresh_stage2 = iou_thresh_stage2
        self.reid_weight = reid_weight
        self.ocm_weight  = ocm_weight

        self.tracked_stracks: list[DOCSTrack] = []
        self.lost_stracks:    list[DOCSTrack] = []
        self.removed_stracks: list[DOCSTrack] = []
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
        STrack.multi_predict(self.tracked_stracks + self.lost_stracks)

        # Stage 1: high dets vs active (OCM + IoU + Re-ID)
        (m1, unm_t1, unm_d1) = self._associate_full(
            high_dets, self.tracked_stracks, self.match_thresh
        )
        for t_idx, d_idx in m1:
            track = self.tracked_stracks[t_idx]
            det   = high_dets[d_idx]
            if track.time_since_update > 0:
                self._ocr_reinit(track, det.bbox)
            track.update(det.bbox, det.score, None)
            track.push_obs(det.bbox)
            if det.embedding is not None:
                track.update_ema(det.embedding)
        unmatched_active = [self.tracked_stracks[i] for i in unm_t1]

        # Stage 2: low dets vs unmatched + lost (IoU only)
        candidate_s2 = unmatched_active + self.lost_stracks
        (m2, unm_s2, _) = self._associate_iou(
            low_dets, candidate_s2, self.iou_thresh_stage2
        )
        matched_s2 = {t for t, _ in m2}
        for t_idx, d_idx in m2:
            track = candidate_s2[t_idx]
            det   = low_dets[d_idx]
            if track.time_since_update > 0:
                self._ocr_reinit(track, det.bbox)
            track.update(det.bbox, det.score, None)
            track.push_obs(det.bbox)
        for i, track in enumerate(candidate_s2):
            if i not in matched_s2:
                if track.state != TrackState.Lost:
                    track.mark_lost()
                else:
                    track.increment_age()

        # New tentative tracks
        for d_idx in unm_d1:
            det = high_dets[d_idx]
            if det.score < self.track_thresh:
                continue
            t = DOCSTrack(det.bbox, det.score)
            t.activate(self.frame_id)
            if det.embedding is not None:
                t.update_ema(det.embedding)
            self.tracked_stracks.append(t)

        # Rebuild
        seen: set[int] = set()
        still_lost: list[DOCSTrack] = []
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
    def _ocr_reinit(self, track: DOCSTrack, new_bbox: np.ndarray) -> None:
        last = track.last_obs()
        last_xyah = STrack.tlbr_to_xyah(last)
        new_xyah  = STrack.tlbr_to_xyah(new_bbox)
        vel = new_xyah - last_xyah
        mean, cov = _kf.initiate(last_xyah)
        mean[4:8] = vel
        track.mean, track.covariance = mean, cov

    def _ocm_penalty(
        self, tracks: list[DOCSTrack], det_boxes: np.ndarray
    ) -> np.ndarray:
        M, N = len(tracks), det_boxes.shape[0]
        penalty = np.zeros((M, N), dtype=np.float64)
        for i, t in enumerate(tracks):
            if len(t.obs_history) < 1:
                continue
            last = t.last_obs()
            lx = (last[0]+last[2])/2
            ly = (last[1]+last[3])/2
            v_t = t.obs_velocity()
            for j in range(N):
                dx = (det_boxes[j,0]+det_boxes[j,2])/2 - lx
                dy = (det_boxes[j,1]+det_boxes[j,3])/2 - ly
                v_d = np.array([dx, dy])
                diff  = np.linalg.norm(v_d - v_t)
                scale = max(np.linalg.norm(v_t), np.linalg.norm(v_d), 1.0)
                penalty[i, j] = min(diff / scale, 1.0)
        return penalty

    def _associate_full(
        self, detections: list[Detection], tracks: list[DOCSTrack], thresh: float
    ) -> tuple:
        if not tracks or not detections:
            return [], list(range(len(tracks))), list(range(len(detections)))
        track_boxes = np.array([t.tlbr for t in tracks])
        det_boxes   = np.array([d.bbox for d in detections])
        iou_d = 1.0 - iou_matrix(track_boxes, det_boxes)
        ocm   = self._ocm_penalty(tracks, det_boxes)

        t_embs = [t.reid_embedding for t in tracks]
        d_embs = [d.embedding      for d in detections]
        if all(e is not None for e in t_embs) and all(e is not None for e in d_embs):
            reid_d = cosine_distance_matrix(np.stack(t_embs), np.stack(d_embs))
            iou_w  = 1.0 - self.reid_weight - self.ocm_weight
            cost   = iou_w * iou_d + self.reid_weight * reid_d + self.ocm_weight * ocm
        else:
            iou_w = 1.0 - self.ocm_weight
            cost  = iou_w * iou_d + self.ocm_weight * ocm
        return linear_assignment(cost, thresh)

    def _associate_iou(
        self, detections: list[Detection], tracks: list[STrack], thresh: float
    ) -> tuple:
        if not tracks or not detections:
            return [], list(range(len(tracks))), list(range(len(detections)))
        cost = 1.0 - iou_matrix(
            np.array([t.tlbr for t in tracks]),
            np.array([d.bbox for d in detections]),
        )
        return linear_assignment(cost, thresh)
