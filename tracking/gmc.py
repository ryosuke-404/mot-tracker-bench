"""
Global Motion Compensation (GMC) for BoT-SORT.

Estimates inter-frame camera motion using sparse optical flow (ECC or ORB)
and applies the affine warp to predicted Kalman track positions, compensating
for camera movement on a moving bus.

Two backends:
  - "orb"   Fast ORB keypoint matching (default, CPU-friendly)
  - "ecc"   Enhanced Correlation Coefficient (more accurate, slower)
  - "none"  Disabled (falls back to ByteTrack behaviour)
"""

from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class GMC:
    """
    Estimates camera motion between consecutive frames and returns
    a 2x3 affine warp matrix H that maps prev frame coords → current frame coords.
    """

    def __init__(self, method: str = "orb", downscale: int = 4) -> None:
        """
        Args:
            method     "orb" | "ecc" | "none"
            downscale  processing scale factor (4 = quarter resolution for speed)
        """
        self.method = method
        self.downscale = max(1, downscale)

        self._prev_frame_gray: Optional[np.ndarray] = None
        self._prev_keypoints: Optional[list] = None
        self._prev_descriptors: Optional[np.ndarray] = None

        if method == "orb":
            self._detector = cv2.FastFeatureDetector_create(20)
            self._extractor = cv2.ORB_create()
            self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        elif method == "ecc":
            self._warp_mode = cv2.MOTION_EUCLIDEAN
            self._num_iter = 50
            self._term_eps = 1e-4
        elif method == "none":
            pass
        else:
            raise ValueError(f"Unknown GMC method: {method!r}")

    # ------------------------------------------------------------------
    def apply(self, frame: np.ndarray) -> np.ndarray:
        """
        Estimate camera motion from previous frame.

        Returns:
            H  shape (2, 3) affine matrix.
               Identity matrix if this is the first frame or estimation fails.
        """
        identity = np.eye(2, 3, dtype=np.float32)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.downscale > 1:
            h, w = gray.shape
            gray = cv2.resize(gray, (w // self.downscale, h // self.downscale))

        if self._prev_frame_gray is None:
            self._prev_frame_gray = gray
            if self.method == "orb":
                kps = self._detector.detect(gray)
                kps, descs = self._extractor.compute(gray, kps)
                self._prev_keypoints = kps
                self._prev_descriptors = descs
            return identity

        if self.method == "orb":
            H = self._apply_orb(gray)
        elif self.method == "ecc":
            H = self._apply_ecc(gray)
        else:
            H = identity

        self._prev_frame_gray = gray
        return H

    # ------------------------------------------------------------------
    def _apply_orb(self, gray: np.ndarray) -> np.ndarray:
        identity = np.eye(2, 3, dtype=np.float32)
        try:
            kps = self._detector.detect(gray)
            kps, descs = self._extractor.compute(gray, kps)

            if (descs is None or self._prev_descriptors is None
                    or len(kps) < 8 or len(self._prev_keypoints) < 8):
                self._prev_keypoints = kps
                self._prev_descriptors = descs
                return identity

            matches = self._matcher.knnMatch(self._prev_descriptors, descs, k=2)
            good = [m for m, n in matches if m.distance < 0.7 * n.distance]

            if len(good) < 4:
                self._prev_keypoints = kps
                self._prev_descriptors = descs
                return identity

            prev_pts = np.float32([self._prev_keypoints[m.queryIdx].pt for m in good])
            curr_pts = np.float32([kps[m.trainIdx].pt for m in good])

            H, inliers = cv2.estimateAffinePartial2D(
                prev_pts, curr_pts, method=cv2.RANSAC, ransacReprojThreshold=3.0
            )

            self._prev_keypoints = kps
            self._prev_descriptors = descs

            if H is None:
                return identity

            # Scale translation back to original resolution
            H[0, 2] *= self.downscale
            H[1, 2] *= self.downscale
            return H.astype(np.float32)

        except Exception as e:
            logger.debug("GMC ORB failed: %s", e)
            return identity

    def _apply_ecc(self, gray: np.ndarray) -> np.ndarray:
        identity = np.eye(2, 3, dtype=np.float32)
        try:
            H = np.eye(2, 3, dtype=np.float32)
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                        self._num_iter, self._term_eps)
            _, H = cv2.findTransformECC(
                self._prev_frame_gray, gray, H,
                self._warp_mode, criteria, None, 1
            )
            H[0, 2] *= self.downscale
            H[1, 2] *= self.downscale
            return H
        except Exception as e:
            logger.debug("GMC ECC failed: %s", e)
            return identity

    # ------------------------------------------------------------------
    def apply_to_tracks(
        self, H: np.ndarray, tracks
    ) -> None:
        """
        Warp Kalman mean positions of all tracks using the affine H.
        Modifies track.mean in-place.
        """
        if np.allclose(H, np.eye(2, 3)):
            return

        for track in tracks:
            if track.mean is None:
                continue
            cx, cy = track.mean[0], track.mean[1]
            # Apply affine: [cx', cy'] = H @ [cx, cy, 1]^T
            new_cx = H[0, 0] * cx + H[0, 1] * cy + H[0, 2]
            new_cy = H[1, 0] * cx + H[1, 1] * cy + H[1, 2]
            # Also warp velocity (rotation part only, no translation)
            vx, vy = track.mean[4], track.mean[5]
            new_vx = H[0, 0] * vx + H[0, 1] * vy
            new_vy = H[1, 0] * vx + H[1, 1] * vy
            track.mean[0] = new_cx
            track.mean[1] = new_cy
            track.mean[4] = new_vx
            track.mean[5] = new_vy

    def reset(self) -> None:
        self._prev_frame_gray = None
        self._prev_keypoints = None
        self._prev_descriptors = None
