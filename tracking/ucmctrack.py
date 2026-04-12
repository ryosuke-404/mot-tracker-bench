"""
UCMCTrack: Multi-Object Tracking with Uniform Camera Motion Compensation
Yi et al., AAAI 2024 — MIT License

Core idea: instead of matching bounding boxes in image space (pixels),
project the bbox *centers* to a ground-plane coordinate system using a
perspective homography H, then match in that 2-D ground plane.

Why this helps for bus cameras:
  - Two persons at different depths have different pixel heights but the
    same ground-plane separation — ground-plane distance is a better
    metric for "are these the same person?"
  - Camera tilt / slight movement affects image coordinates strongly but
    ground-plane coordinates minimally (if H is reasonable).

Ground-plane Kalman:
  State: [gx, gy, vgx, vgy]  (ground-plane center + velocity)
  Measurement: [gx, gy]

Homography estimation:
  Without calibration data we estimate H from 4 control points that map
  typical image regions to a canonical front-facing plane.  The defaults
  correspond to a typical bus interior camera mounted at ~2 m height,
  ~30° downward tilt.  Pass your own H_world as a 3×3 matrix to override.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.linalg

from tracking.track import STrack, TrackState
from tracking.bytetrack import Detection, iou_matrix, linear_assignment


# ---------------------------------------------------------------------------
# Default perspective transform (image → ground plane)
# Tuned for 2560×1440, bus interior, camera ~30° downward tilt
# ---------------------------------------------------------------------------

def default_homography(img_w: int = 2560, img_h: int = 1440) -> np.ndarray:
    """
    Estimate a simple ground-plane homography from typical bus interior geometry.
    Maps image (x, y) → approximate ground-plane (u, v) in cm.
    """
    # Source: 4 image-space points (proportional)
    src = np.float32([
        [0.1 * img_w, 0.5 * img_h],   # bottom-left region
        [0.9 * img_w, 0.5 * img_h],   # bottom-right region
        [0.8 * img_w, 0.9 * img_h],   # near-center right
        [0.2 * img_w, 0.9 * img_h],   # near-center left
    ])
    # Destination: metric ground plane (arbitrary scale, cm)
    dst = np.float32([
        [0,   400],
        [600, 400],
        [500, 0],
        [100, 0],
    ])
    H, _ = cv2.findHomography(src, dst)
    return H if H is not None else np.eye(3, dtype=np.float32)


# ---------------------------------------------------------------------------
# Ground-plane Kalman filter (4-state: gx, gy, vgx, vgy)
# ---------------------------------------------------------------------------

class GroundKalman:
    """Simple 2-D constant-velocity Kalman filter in ground-plane coordinates."""

    def __init__(self) -> None:
        dt = 1.0
        self.F = np.array([[1, 0, dt, 0],
                            [0, 1, 0, dt],
                            [0, 0, 1,  0],
                            [0, 0, 0,  1]], dtype=np.float64)
        self.H = np.array([[1, 0, 0, 0],
                            [0, 1, 0, 0]], dtype=np.float64)
        self.Q = np.diag([10.0, 10.0, 5.0, 5.0])   # process noise (cm)
        self.R = np.diag([25.0, 25.0])               # measurement noise (cm)

    def initiate(self, gxy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mean = np.r_[gxy, np.zeros(2)]
        P    = np.diag([50.0, 50.0, 100.0, 100.0])
        return mean, P

    def predict(self, mean: np.ndarray, P: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mean = self.F @ mean
        P    = self.F @ P @ self.F.T + self.Q
        return mean, P

    def update(
        self, mean: np.ndarray, P: np.ndarray, z: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        S  = self.H @ P @ self.H.T + self.R
        K  = P @ self.H.T @ np.linalg.inv(S)
        mean = mean + K @ (z - self.H @ mean)
        P    = (np.eye(4) - K @ self.H) @ P
        return mean, P

    def gxy(self, mean: np.ndarray) -> np.ndarray:
        return mean[:2]


_gk = GroundKalman()


# ---------------------------------------------------------------------------
# Per-track ground-plane state
# ---------------------------------------------------------------------------

@dataclass
class GPState:
    mean: np.ndarray   # (4,)
    cov:  np.ndarray   # (4,4)

    def gxy(self) -> np.ndarray:
        return self.mean[:2]


# ---------------------------------------------------------------------------
# UCMCTracker
# ---------------------------------------------------------------------------

class UCMCTracker:
    """
    UCMCTrack multi-object tracker.

    Drop-in replacement for ByteTracker (same update() signature).

    Matching is performed in ground-plane (metric) coordinates.
    Euclidean distance in ground-plane replaces IoU distance.
    """

    def __init__(
        self,
        track_thresh: float = 0.45,
        track_buffer: int = 90,
        match_thresh_gp: float = 150.0,   # max ground-plane distance (cm) for match
        min_hits: int = 3,
        img_w: int = 2560,
        img_h: int = 1440,
        H_world: np.ndarray | None = None,
    ) -> None:
        self.track_thresh    = track_thresh
        self.track_buffer    = track_buffer
        self.match_thresh_gp = match_thresh_gp
        self.min_hits        = min_hits

        global cv2
        import cv2 as _cv2
        cv2 = _cv2

        self._H = H_world if H_world is not None else default_homography(img_w, img_h)
        self._H_inv = np.linalg.inv(self._H)

        self.tracked_stracks: list[STrack] = []
        self.lost_stracks:    list[STrack] = []
        self.removed_stracks: list[STrack] = []
        self._gp_states: dict[int, GPState] = {}   # track_id → GP Kalman state
        self.frame_id = 0

    def reset(self) -> None:
        self.tracked_stracks.clear()
        self.lost_stracks.clear()
        self.removed_stracks.clear()
        self._gp_states.clear()
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

        # Predict Kalman (image-space) + ground-plane Kalman
        all_tracks = self.tracked_stracks + self.lost_stracks
        STrack.multi_predict(all_tracks)
        for t in all_tracks:
            if t.track_id in self._gp_states:
                gps = self._gp_states[t.track_id]
                gps.mean, gps.cov = _gk.predict(gps.mean, gps.cov)

        # Convert detections to ground-plane coords
        high_gxy = [self._img_to_gp(d.bbox) for d in high_dets]
        low_gxy  = [self._img_to_gp(d.bbox) for d in low_dets]

        # Stage 1: high dets vs active (ground-plane Euclidean)
        (m1, unm_t1, unm_d1) = self._associate_gp(
            self.tracked_stracks, high_gxy, self.match_thresh_gp
        )
        for t_idx, d_idx in m1:
            track = self.tracked_stracks[t_idx]
            det   = high_dets[d_idx]
            track.update(det.bbox, det.score, det.embedding)
            self._update_gp(track, high_gxy[d_idx])

        unmatched_active = [self.tracked_stracks[i] for i in unm_t1]

        # Stage 2: low dets vs unmatched + lost (ground-plane)
        candidate_s2 = unmatched_active + self.lost_stracks
        (m2, unm_s2, _) = self._associate_gp(
            candidate_s2, low_gxy, self.match_thresh_gp * 1.5
        )
        matched_s2 = {t for t, _ in m2}
        for t_idx, d_idx in m2:
            candidate_s2[t_idx].update(low_dets[d_idx].bbox, low_dets[d_idx].score, None)
            self._update_gp(candidate_s2[t_idx], low_gxy[d_idx])
        for i, track in enumerate(candidate_s2):
            if i not in matched_s2:
                if track.state != TrackState.Lost:
                    track.mark_lost()
                else:
                    track.increment_age()

        # New tracks
        for d_idx in unm_d1:
            det = high_dets[d_idx]
            if det.score < self.track_thresh:
                continue
            t = STrack(det.bbox, det.score)
            t.activate(self.frame_id)
            if det.embedding is not None:
                t.reid_embedding = det.embedding
                t.embedding_history.append(det.embedding)
            m, c = _gk.initiate(high_gxy[d_idx])
            self._gp_states[t.track_id] = GPState(mean=m, cov=c)
            self.tracked_stracks.append(t)

        # Rebuild
        seen: set[int] = set()
        still_lost: list[STrack] = []
        for track in (self.lost_stracks
                      + [t for t in unmatched_active if t.state == TrackState.Lost]):
            if track.time_since_update > self.track_buffer:
                track.mark_removed()
                self.removed_stracks.append(track)
                self._gp_states.pop(track.track_id, None)
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

    def _img_to_gp(self, bbox: np.ndarray) -> np.ndarray:
        """Project bbox foot-point (bottom-center) to ground plane."""
        cx = (bbox[0] + bbox[2]) / 2
        cy = bbox[3]                    # bottom of bbox (foot point)
        pt = np.array([[[cx, cy]]], dtype=np.float32)
        gp = cv2.perspectiveTransform(pt, self._H)
        return gp[0, 0].astype(np.float64)   # (2,)

    def _update_gp(self, track: STrack, gxy: np.ndarray) -> None:
        if track.track_id in self._gp_states:
            gps = self._gp_states[track.track_id]
            gps.mean, gps.cov = _gk.update(gps.mean, gps.cov, gxy)
        else:
            m, c = _gk.initiate(gxy)
            self._gp_states[track.track_id] = GPState(mean=m, cov=c)

    def _associate_gp(
        self,
        tracks: list[STrack],
        det_gxy: list[np.ndarray],
        thresh: float,
    ) -> tuple:
        """Match by Euclidean distance in ground plane."""
        if not tracks or not det_gxy:
            return [], list(range(len(tracks))), list(range(len(det_gxy)))

        M, N = len(tracks), len(det_gxy)
        cost = np.full((M, N), fill_value=thresh + 1.0, dtype=np.float64)
        for i, track in enumerate(tracks):
            gps = self._gp_states.get(track.track_id)
            if gps is None:
                # Fallback: project current tlbr bottom-center
                b  = track.tlbr
                gxy_t = self._img_to_gp(b)
            else:
                gxy_t = gps.gxy()
            for j, gxy_d in enumerate(det_gxy):
                cost[i, j] = float(np.linalg.norm(gxy_t - gxy_d))

        # Normalize cost to [0, 1] for linear_assignment with thresh in [0,1]
        cost_norm = cost / (thresh + 1e-6)
        return linear_assignment(cost_norm, 1.0)
