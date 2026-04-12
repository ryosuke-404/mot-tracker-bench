"""
Re-ID gallery for long-term identity maintenance.

The gallery stores appearance embeddings (L2-normalized feature vectors) for
each confirmed track.  When ByteTrack creates a new STrack but the passenger
was actually lost briefly (occluded, or track buffer expired), the gallery
query can resurrect the original track_id instead of assigning a new one.

This is the key mechanism that prevents double-counting passengers:
  - A passenger sits down → Kalman prediction fails → track goes Lost → Removed
  - Passenger stands up to exit → new detection → gallery match → same track_id
  - OD tracker sees same track_id → recognises it already has a boarding record
"""

from __future__ import annotations

import logging
import threading
from collections import defaultdict, deque

import numpy as np

logger = logging.getLogger(__name__)


class ReIDGallery:
    """
    Rolling-window embedding gallery with cosine-similarity queries.

    Thread-safe: gallery updates and queries may come from the pipeline thread
    while pruning runs in a background thread.
    """

    def __init__(
        self,
        max_gallery_size: int = 500,
        similarity_thresh: float = 0.75,
        max_embeddings_per_id: int = 10,
    ) -> None:
        self.similarity_thresh = similarity_thresh
        self.max_gallery_size = max_gallery_size
        self.max_embeddings_per_id = max_embeddings_per_id

        # track_id → deque of L2-normalized embeddings
        self._gallery: dict[int, deque[np.ndarray]] = defaultdict(
            lambda: deque(maxlen=max_embeddings_per_id)
        )
        # track_id → frame_id of last update (for age-based pruning)
        self._last_seen: dict[int, int] = {}
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    def add_embedding(
        self, track_id: int, embedding: np.ndarray, frame_id: int = 0
    ) -> None:
        """
        Add a new embedding for a track to the gallery.

        Args:
            track_id   confirmed STrack id
            embedding  L2-normalized feature vector, shape (D,)
            frame_id   current frame index (used for pruning)
        """
        with self._lock:
            if len(self._gallery) >= self.max_gallery_size and track_id not in self._gallery:
                # Evict the oldest entry to stay within budget
                oldest_id = min(self._last_seen, key=self._last_seen.get)
                del self._gallery[oldest_id]
                del self._last_seen[oldest_id]
            self._gallery[track_id].append(embedding)
            self._last_seen[track_id] = frame_id

    # ------------------------------------------------------------------
    def query(
        self,
        probe_embedding: np.ndarray,
        exclude_ids: set[int] | None = None,
    ) -> tuple[int | None, float]:
        """
        Find the best gallery match for a probe embedding.

        Args:
            probe_embedding   L2-normalized feature vector, shape (D,)
            exclude_ids       track_ids to skip (currently active tracks)

        Returns:
            (track_id, similarity)  or  (None, 0.0) if no match above thresh
        """
        exclude_ids = exclude_ids or set()

        best_id: int | None = None
        best_sim: float = -1.0

        with self._lock:
            for track_id, embeddings in self._gallery.items():
                if track_id in exclude_ids:
                    continue
                if not embeddings:
                    continue

                rep = self._get_representative(track_id)
                sim = float(np.dot(probe_embedding, rep))  # cosine sim (both L2-norm)

                if sim > best_sim:
                    best_sim = sim
                    best_id = track_id

        if best_sim >= self.similarity_thresh:
            return best_id, best_sim
        return None, 0.0

    def query_batch(
        self,
        probe_embeddings: np.ndarray,
        exclude_ids: set[int] | None = None,
    ) -> list[tuple[int | None, float]]:
        """
        Batch query: find best gallery match for each probe.

        Args:
            probe_embeddings  shape (N, D)
            exclude_ids       track_ids to skip

        Returns:
            list of (track_id, similarity) per probe
        """
        exclude_ids = exclude_ids or set()
        results: list[tuple[int | None, float]] = []

        with self._lock:
            # Build gallery matrix for all non-excluded ids
            gallery_ids: list[int] = []
            gallery_reps: list[np.ndarray] = []
            for tid, embeddings in self._gallery.items():
                if tid in exclude_ids or not embeddings:
                    continue
                gallery_ids.append(tid)
                gallery_reps.append(self._get_representative(tid))

            if not gallery_ids:
                return [(None, 0.0)] * len(probe_embeddings)

            G = np.stack(gallery_reps, axis=0)   # (K, D)
            # Cosine similarity matrix: (N, K)
            sims = probe_embeddings @ G.T

            for row in sims:
                best_k = int(np.argmax(row))
                best_sim = float(row[best_k])
                if best_sim >= self.similarity_thresh:
                    results.append((gallery_ids[best_k], best_sim))
                else:
                    results.append((None, 0.0))

        return results

    # ------------------------------------------------------------------
    def remove_track(self, track_id: int) -> None:
        """Explicitly remove a track from the gallery (e.g., after confirmed exit)."""
        with self._lock:
            self._gallery.pop(track_id, None)
            self._last_seen.pop(track_id, None)

    def prune_old_tracks(
        self, active_ids: set[int], max_age_frames: int, current_frame: int
    ) -> int:
        """
        Remove gallery entries for tracks that are both inactive and old.

        Should be called periodically (e.g., every 100 frames) rather than
        every frame to avoid excessive overhead.

        Returns:
            number of entries pruned
        """
        pruned = 0
        with self._lock:
            to_remove = [
                tid
                for tid, last_f in self._last_seen.items()
                if tid not in active_ids
                and (current_frame - last_f) > max_age_frames
            ]
            for tid in to_remove:
                del self._gallery[tid]
                del self._last_seen[tid]
                pruned += 1

        if pruned:
            logger.debug("Gallery pruned %d stale entries (frame %d)", pruned, current_frame)
        return pruned

    def size(self) -> int:
        with self._lock:
            return len(self._gallery)

    # ------------------------------------------------------------------
    def _get_representative(self, track_id: int) -> np.ndarray:
        """
        Return the representative embedding for a track (mean of gallery,
        then L2-renormalized).
        """
        embeds = np.stack(list(self._gallery[track_id]), axis=0)  # (K, D)
        mean_embed = embeds.mean(axis=0)
        norm = np.linalg.norm(mean_embed)
        return mean_embed / max(norm, 1e-12)
