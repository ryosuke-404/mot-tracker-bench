"""
Post-processing to reduce ID fragmentation after tracking.

Strategy D — Appearance Merge:
    Merge non-overlapping tracks whose mean-embedding cosine similarity
    exceeds a threshold. Works without knowing GT person count.

Strategy E — Cluster Re-assign:
    Force exactly k IDs using agglomerative clustering on track embeddings.
    Requires knowing GT person count (k).

Typical usage (offline, after all frames are processed):
    mean_embeds = compute_mean_embeddings(track_embeddings)
    id_map      = merge_by_appearance(track_birth, track_death, mean_embeds)
    id_map      = cluster_reassign(track_birth, mean_embeds, n_clusters=7)
    new_birth, new_death = apply_id_map(track_birth, track_death, id_map)
"""

from __future__ import annotations

from itertools import combinations

import numpy as np


# ---------------------------------------------------------------------------

def compute_mean_embeddings(
    track_embeddings: dict[int, list[np.ndarray]],
) -> dict[int, np.ndarray]:
    """L2-normalised mean embedding per track ID."""
    result: dict[int, np.ndarray] = {}
    for tid, embeds in track_embeddings.items():
        if not embeds:
            continue
        arr = np.stack(embeds)               # (K, D)
        mean = arr.mean(axis=0)              # (D,)
        norm = np.linalg.norm(mean)
        result[tid] = mean / norm if norm > 1e-12 else mean
    return result


# ---------------------------------------------------------------------------
# Strategy D — greedy appearance merge
# ---------------------------------------------------------------------------

def merge_by_appearance(
    track_birth: dict[int, int],
    track_death: dict[int, int],
    track_mean_embeds: dict[int, np.ndarray],
    sim_thresh: float = 0.65,
    gap_tolerance: int = 5,
) -> dict[int, int]:
    """
    Greedily merge non-overlapping tracks whose appearance similarity
    exceeds sim_thresh.

    Two tracks are considered non-overlapping when neither is active
    while the other is (with gap_tolerance frames of slack for brief
    detector dropout).

    Returns:
        id_map   dict mapping old_track_id → new_track_id
                 (new_id = earliest-born track in the merged group)
    """
    tids = [t for t in track_mean_embeds if t in track_birth]
    if not tids:
        return {}

    # Sort by birth so we naturally keep the earlier ID as root
    tids.sort(key=lambda t: track_birth[t])

    # Union-Find ----------------------------------------------------------
    parent: dict[int, int] = {t: t for t in tids}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        # Keep earlier-born track as root
        if track_birth.get(ra, 0) <= track_birth.get(rb, 0):
            parent[rb] = ra
        else:
            parent[ra] = rb

    # Candidate pairs: time-non-overlapping + similar appearance ----------
    for a, b in combinations(tids, 2):
        a_s = track_birth.get(a, 0)
        a_e = track_death.get(a, 0)
        b_s = track_birth.get(b, 0)
        b_e = track_death.get(b, 0)

        # Check time non-overlap (with tolerance)
        if not (a_e + gap_tolerance < b_s or b_e + gap_tolerance < a_s):
            continue

        sim = float(np.dot(track_mean_embeds[a], track_mean_embeds[b]))
        if sim >= sim_thresh:
            union(a, b)

    return {t: find(t) for t in tids}


# ---------------------------------------------------------------------------
# Strategy E — agglomerative cluster re-assign
# ---------------------------------------------------------------------------

def cluster_reassign(
    track_birth: dict[int, int],
    track_mean_embeds: dict[int, np.ndarray],
    n_clusters: int,
) -> dict[int, int]:
    """
    Re-assign all track IDs using agglomerative clustering with n_clusters
    clusters (typically = known GT person count).

    Each cluster is represented by the track with the earliest birth time.

    Returns:
        id_map   dict mapping old_track_id → new_track_id
    """
    from sklearn.cluster import AgglomerativeClustering

    tids = list(track_mean_embeds.keys())
    n = len(tids)
    if n == 0:
        return {}
    if n <= n_clusters:
        return {t: t for t in tids}

    embeds = np.stack([track_mean_embeds[t] for t in tids])   # (N, D)

    labels: np.ndarray = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric="cosine",
        linkage="average",
    ).fit_predict(embeds)

    # Representative = earliest-born track in each cluster
    cluster_rep: dict[int, int] = {}
    for tid, label in zip(tids, labels.tolist()):
        label = int(label)
        if label not in cluster_rep:
            cluster_rep[label] = tid
        elif track_birth.get(tid, 0) < track_birth.get(cluster_rep[label], 0):
            cluster_rep[label] = tid

    return {tid: cluster_rep[int(label)] for tid, label in zip(tids, labels.tolist())}


# ---------------------------------------------------------------------------
# Apply id_map to track timeline dicts
# ---------------------------------------------------------------------------

def apply_id_map(
    track_birth: dict[int, int],
    track_death: dict[int, int],
    id_map: dict[int, int],
) -> tuple[dict[int, int], dict[int, int]]:
    """
    Collapse track_birth / track_death according to id_map.
    The merged track spans min(births) … max(deaths) of its group.
    """
    new_birth: dict[int, int] = {}
    new_death: dict[int, int] = {}
    for old_id, new_id in id_map.items():
        b = track_birth.get(old_id, 0)
        d = track_death.get(old_id, 0)
        new_birth[new_id] = min(new_birth.get(new_id, b), b)
        new_death[new_id] = max(new_death.get(new_id, d), d)
    # Pass through tracks not in id_map
    for tid in track_birth:
        if tid not in id_map:
            new_birth[tid] = track_birth[tid]
            new_death[tid] = track_death.get(tid, track_birth[tid])
    return new_birth, new_death


# ---------------------------------------------------------------------------
# Compose two id_maps (apply map2 on top of map1's output)
# ---------------------------------------------------------------------------

def compose_maps(
    map1: dict[int, int],
    map2: dict[int, int],
    all_tids: list[int],
) -> dict[int, int]:
    result: dict[int, int] = {}
    for tid in all_tids:
        intermediate = map1.get(tid, tid)
        result[tid] = map2.get(intermediate, intermediate)
    return result
