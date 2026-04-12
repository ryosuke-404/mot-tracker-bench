"""
StrongSORT: Making DeepSORT Great Again
Du et al., TMM 2023 — Apache 2.0 License

Key improvements over DeepSORT:
  1. NSA Kalman (Noise Scale Adaptive Kalman Filter)
     Scale measurement noise R by (1 - confidence):
       R' = R * (1 - conf)
     High-confidence detections reduce R, making Kalman trust the
     measurement more and update the state more aggressively.

  2. EMA (Exponential Moving Average) appearance features
     Track embedding updated as:
       f_track = alpha * f_track + (1 - alpha) * f_det
     Smoother than per-frame replacement; reduces noise in appearance cost.

  3. GMC (Global Motion Compensation)
     Same ORB-based affine warp as BoT-SORT to correct Kalman predictions
     for camera motion — critical for bus/vehicle cameras.

  4. Improved Re-ID model
     In the original paper, OSNet replaces the shallow MARS network.
     Our extractor already uses OSNet, so this is satisfied.

Post-processing (AFLink + GSI) is omitted as it requires offline data.
"""

from __future__ import annotations

import numpy as np
import scipy.linalg

from tracking.track import STrack, TrackState
from tracking.bytetrack import Detection, iou_matrix, cosine_distance_matrix, linear_assignment
from tracking.kalman_filter import KalmanFilter
from tracking.gmc import GMC

_base_kalman = KalmanFilter()

# Mahalanobis gate (chi-sq 95%, DOF=4)
_MAHA_GATE = KalmanFilter.chi2inv95[4]


# ---------------------------------------------------------------------------
# NSA Kalman update (confidence-scaled measurement noise)
# ---------------------------------------------------------------------------

def nsa_kalman_update(
    mean: np.ndarray,
    covariance: np.ndarray,
    measurement: np.ndarray,
    confidence: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Kalman update with NSA: measurement noise R is scaled by (1 - conf).
    confidence ∈ [0, 1]; higher confidence → smaller R → stronger correction.
    """
    h = mean[3]
    std = [
        _base_kalman._std_pos * h,
        _base_kalman._std_pos * h,
        1e-1,
        _base_kalman._std_pos * h,
    ]
    # NSA scaling
    scale = 1.0 - confidence
    R = np.diag(np.square(np.array(std) * scale + 1e-6))

    H  = _base_kalman._H
    projected_mean = H @ mean
    projected_cov  = H @ covariance @ H.T + R

    chol, lower = scipy.linalg.cho_factor(projected_cov, lower=True, check_finite=False)
    kalman_gain  = scipy.linalg.cho_solve(
        (chol, lower), H @ covariance, check_finite=False
    ).T

    innovation = measurement - projected_mean
    new_mean = mean + innovation @ kalman_gain.T
    new_cov  = covariance - kalman_gain @ projected_cov @ kalman_gain.T
    return new_mean, new_cov


# ---------------------------------------------------------------------------
# StrongSORT track with EMA embedding
# ---------------------------------------------------------------------------

class SSSTrack(STrack):
    """Extends STrack with EMA appearance feature blending."""

    EMA_ALPHA = 0.9    # blend factor: 0.9 = heavy weight on historical embedding

    def update_ema(self, new_embedding: np.ndarray) -> None:
        """Blend new embedding into track's EMA embedding."""
        if self.reid_embedding is None:
            self.reid_embedding = new_embedding.copy()
        else:
            self.reid_embedding = (
                self.EMA_ALPHA * self.reid_embedding
                + (1.0 - self.EMA_ALPHA) * new_embedding
            )
            # Re-normalize after EMA blend
            norm = np.linalg.norm(self.reid_embedding)
            if norm > 1e-12:
                self.reid_embedding /= norm
        self.embedding_history.append(self.reid_embedding.copy())

    def update_nsa(
        self,
        new_tlbr: np.ndarray,
        score: float,
        embedding: np.ndarray | None = None,
    ) -> None:
        """Update with NSA Kalman (confidence-scaled noise)."""
        self._tlbr = new_tlbr.copy()
        self.score = score
        self.mean, self.covariance = nsa_kalman_update(
            self.mean, self.covariance,
            self.tlbr_to_xyah(new_tlbr), score
        )
        if embedding is not None:
            self.update_ema(embedding)
        self.hits += 1
        self.age  += 1
        self.time_since_update = 0
        if self.state == TrackState.Tentative and self.hits >= 3:
            self.state = TrackState.Confirmed
        elif self.state == TrackState.Lost:
            self.state = TrackState.Confirmed


# ---------------------------------------------------------------------------
# StrongSORT Tracker
# ---------------------------------------------------------------------------

class StrongSORTTracker:
    """
    StrongSORT multi-object tracker.

    Drop-in replacement for ByteTracker (same update() signature).

    Matching pipeline mirrors DeepSORT cascade but with:
      - NSA Kalman update (confidence-scaled R)
      - EMA appearance blending
      - GMC camera-motion correction
    """

    def __init__(
        self,
        track_thresh: float = 0.45,
        track_buffer: int = 90,
        min_hits: int = 3,
        max_cosine_dist: float = 0.25,
        nn_budget: int = 100,
        iou_thresh_fallback: float = 0.70,
        iou_thresh_stage2: float = 0.45,
        gmc_method: str = "orb",
    ) -> None:
        self.track_thresh        = track_thresh
        self.track_buffer        = track_buffer
        self.min_hits            = min_hits
        self.max_cosine_dist     = max_cosine_dist
        self.nn_budget           = nn_budget
        self.iou_thresh_fallback = iou_thresh_fallback
        self.iou_thresh_stage2   = iou_thresh_stage2

        self.tracked_stracks: list[SSSTrack] = []
        self.lost_stracks:    list[SSSTrack] = []
        self.removed_stracks: list[SSSTrack] = []
        self.frame_id = 0
        self.gmc = GMC(method=gmc_method)

    # ------------------------------------------------------------------
    def reset(self) -> None:
        self.tracked_stracks.clear()
        self.lost_stracks.clear()
        self.removed_stracks.clear()
        self.frame_id = 0
        self.gmc.reset()
        STrack.reset_id_counter()

    # ------------------------------------------------------------------
    def update(
        self,
        high_dets: list[Detection],
        low_dets:  list[Detection],
        frame: np.ndarray | None = None,
    ) -> list[STrack]:
        self.frame_id += 1

        # Predict + GMC correction
        all_tracks = self.tracked_stracks + self.lost_stracks
        STrack.multi_predict(all_tracks)
        if frame is not None and self.gmc.method != "none":
            H = self.gmc.apply(frame)
            self.gmc.apply_to_tracks(H, all_tracks)

        confirmed = [t for t in self.tracked_stracks if t.state == TrackState.Confirmed]
        tentative = [t for t in self.tracked_stracks if t.state == TrackState.Tentative]

        # ---- Cascade match: confirmed × high dets -----------------------
        (matches_c, unm_confirmed_idx,
         unm_high_after_c) = self._cascade_match(confirmed, high_dets)

        for t_idx, d_idx in matches_c:
            t = confirmed[t_idx]
            d = high_dets[d_idx]
            t.update_nsa(d.bbox, d.score, d.embedding)

        # ---- IoU fallback: remaining confirmed + tentative × remaining high
        remaining_t  = [confirmed[i] for i in unm_confirmed_idx] + tentative
        remaining_d  = [high_dets[i] for i in unm_high_after_c]
        (matches_iou, unm_rt_idx,
         unm_high_idx) = self._iou_match(remaining_t, remaining_d,
                                          self.iou_thresh_fallback)
        for t_idx, d_idx in matches_iou:
            remaining_t[t_idx].update_nsa(
                remaining_d[d_idx].bbox,
                remaining_d[d_idx].score,
                remaining_d[d_idx].embedding,
            )

        unmatched_active = [remaining_t[i] for i in unm_rt_idx]

        # ---- Low-score dets vs unmatched active + lost (IoU only) -------
        candidate_s3 = unmatched_active + self.lost_stracks
        (matches_s3, unm_s3_idx, _) = self._iou_match(
            candidate_s3, low_dets, self.iou_thresh_stage2
        )
        matched_s3 = {t for t, _ in matches_s3}
        for t_idx, d_idx in matches_s3:
            candidate_s3[t_idx].update_nsa(
                low_dets[d_idx].bbox, low_dets[d_idx].score, None
            )
        for i, track in enumerate(candidate_s3):
            if i not in matched_s3:
                if track.state != TrackState.Lost:
                    track.mark_lost()
                else:
                    track.increment_age()

        # ---- New tentative tracks ----------------------------------------
        for d_idx in unm_high_idx:
            det = remaining_d[d_idx]
            if det.score < self.track_thresh:
                continue
            new_track = SSSTrack(det.bbox, det.score)
            new_track.activate(self.frame_id)
            if det.embedding is not None:
                new_track.update_ema(det.embedding)
            self.tracked_stracks.append(new_track)

        # ---- Rebuild lists -----------------------------------------------
        still_lost: list[SSSTrack] = []
        for track in (self.lost_stracks
                      + [t for t in unmatched_active if t.state == TrackState.Lost]):
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

    def _cascade_match(
        self,
        confirmed: list[SSSTrack],
        detections: list[Detection],
    ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        if not confirmed or not detections:
            return [], list(range(len(confirmed))), list(range(len(detections)))

        unmatched_det_idx = list(range(len(detections)))
        unmatched_trk_idx = list(range(len(confirmed)))
        all_matches: list[tuple[int, int]] = []
        has_embeds = all(d.embedding is not None for d in detections)

        age_groups: dict[int, list[int]] = {}
        for i, t in enumerate(confirmed):
            age_groups.setdefault(t.time_since_update, []).append(i)

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
            self._apply_maha_gate(cost,
                                  [confirmed[i] for i in t_indices],
                                  [detections[j] for j in d_indices])

            for local_t, local_d in linear_assignment(cost, self.max_cosine_dist)[0]:
                global_t = t_indices[local_t]
                global_d = d_indices[local_d]
                all_matches.append((global_t, global_d))
                unmatched_trk_idx.remove(global_t)
                unmatched_det_idx.remove(global_d)

        return all_matches, unmatched_trk_idx, unmatched_det_idx

    def _appearance_cost(
        self,
        tracks: list[SSSTrack],
        detections: list[Detection],
        has_embeds: bool,
    ) -> np.ndarray:
        M, N = len(tracks), len(detections)
        cost = np.ones((M, N), dtype=np.float64)
        if not has_embeds:
            return cost
        t_embeds = [t.reid_embedding for t in tracks]
        if all(e is not None for e in t_embeds):
            T = np.stack(t_embeds)
            D = np.stack([d.embedding for d in detections])
            cost = cosine_distance_matrix(T, D)
        return cost

    def _apply_maha_gate(
        self,
        cost: np.ndarray,
        tracks: list[SSSTrack],
        detections: list[Detection],
    ) -> None:
        measurements = np.array([STrack.tlbr_to_xyah(d.bbox) for d in detections])
        for i, track in enumerate(tracks):
            if track.mean is None:
                continue
            dists = _base_kalman.gating_distance(
                track.mean, track.covariance, measurements
            )
            cost[i, dists > _MAHA_GATE] = 1.0

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
