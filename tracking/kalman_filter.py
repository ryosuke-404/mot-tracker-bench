"""
Kalman filter for bounding box motion estimation.

State vector: [cx, cy, a, h, vcx, vcy, va, vh]
  cx, cy  — bounding box center (normalized or pixel)
  a       — aspect ratio (width / height)
  h       — height
  vcx, vcy, va, vh — corresponding velocities

Measurement vector: [cx, cy, a, h]
"""

from __future__ import annotations

import numpy as np
import scipy.linalg


class KalmanFilter:
    """
    Kalman filter adapted from SORT / ByteTrack for axis-aligned bounding box
    tracking with constant-velocity motion model.
    """

    # Chi-squared table for gating (95% confidence, DOF = measurement dim)
    chi2inv95 = {1: 3.8415, 2: 5.9915, 3: 7.8147, 4: 9.4877}

    def __init__(self) -> None:
        ndim = 4          # measurement dimension
        dt = 1.0          # time step (1 frame)

        # State transition matrix F (8x8)
        self._F = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._F[i, ndim + i] = dt

        # Observation matrix H (4x8)
        self._H = np.eye(ndim, 2 * ndim)

        # Noise scale factors
        self._std_pos = 1.0 / 20.0    # position noise relative to height
        self._std_vel = 1.0 / 160.0   # velocity noise relative to height

    # ------------------------------------------------------------------
    def initiate(self, measurement: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Create initial state from first measurement [cx, cy, a, h].

        Returns:
            mean        shape (8,)
            covariance  shape (8, 8)
        """
        mean = np.r_[measurement, np.zeros(4)]
        h = measurement[3]
        std = [
            2 * self._std_pos * h,
            2 * self._std_pos * h,
            1e-2,
            2 * self._std_pos * h,
            10 * self._std_vel * h,
            10 * self._std_vel * h,
            1e-5,
            10 * self._std_vel * h,
        ]
        covariance = np.diag(np.square(std))
        return mean, covariance

    # ------------------------------------------------------------------
    def predict(
        self, mean: np.ndarray, covariance: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Propagate state distribution one time step forward.
        """
        h = mean[3]
        std_pos = [
            self._std_pos * h,
            self._std_pos * h,
            1e-2,
            self._std_pos * h,
        ]
        std_vel = [
            self._std_vel * h,
            self._std_vel * h,
            1e-5,
            self._std_vel * h,
        ]
        Q = np.diag(np.square(np.r_[std_pos, std_vel]))

        mean = self._F @ mean
        covariance = self._F @ covariance @ self._F.T + Q
        return mean, covariance

    # ------------------------------------------------------------------
    def project(
        self, mean: np.ndarray, covariance: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Project state distribution to measurement space.
        """
        h = mean[3]
        std = [
            self._std_pos * h,
            self._std_pos * h,
            1e-1,
            self._std_pos * h,
        ]
        R = np.diag(np.square(std))

        projected_mean = self._H @ mean
        projected_cov = self._H @ covariance @ self._H.T + R
        return projected_mean, projected_cov

    # ------------------------------------------------------------------
    def update(
        self,
        mean: np.ndarray,
        covariance: np.ndarray,
        measurement: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Run correction step given a new measurement.
        """
        projected_mean, projected_cov = self.project(mean, covariance)

        chol, lower = scipy.linalg.cho_factor(
            projected_cov, lower=True, check_finite=False
        )
        # Kalman gain K = P H^T S^{-1}  where S = H P H^T + R
        # Solve S X = H P  →  X shape (4, 8), then K = X^T shape (8, 4)
        kalman_gain = scipy.linalg.cho_solve(
            (chol, lower),
            self._H @ covariance,   # shape (4, 8)
            check_finite=False,
        ).T

        innovation = measurement - projected_mean
        new_mean = mean + innovation @ kalman_gain.T
        new_cov = covariance - kalman_gain @ projected_cov @ kalman_gain.T
        return new_mean, new_cov

    # ------------------------------------------------------------------
    def gating_distance(
        self,
        mean: np.ndarray,
        covariance: np.ndarray,
        measurements: np.ndarray,
        only_position: bool = False,
    ) -> np.ndarray:
        """
        Compute squared Mahalanobis distance between state and measurements.

        Args:
            measurements  shape (N, 4)  — [cx, cy, a, h]
            only_position             — use only (cx, cy) for gating

        Returns:
            distances  shape (N,)
        """
        projected_mean, projected_cov = self.project(mean, covariance)

        if only_position:
            projected_mean = projected_mean[:2]
            projected_cov = projected_cov[:2, :2]
            measurements = measurements[:, :2]

        d = measurements - projected_mean               # (N, dim)
        chol = np.linalg.cholesky(projected_cov)        # lower triangular
        # Solve chol @ z.T = d.T  →  z = (chol^-1 @ d.T).T
        z = scipy.linalg.solve_triangular(
            chol, d.T, lower=True, check_finite=False, overwrite_b=True
        )
        return np.sum(z * z, axis=0)                    # (N,)
