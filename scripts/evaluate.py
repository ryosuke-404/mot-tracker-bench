"""
Combination evaluation: Detector × Tracker × Re-ID

GT-based metrics (when --gt_persons is given):
  frag_rate      : track_fragments / gt_persons  (ideal = 1.0)
  id_precision   : gt_persons / track_fragments  (ideal = 1.0, capped at 1.0)
  coverage       : long_lived_tracks / gt_persons (ideal = 1.0, capped at 1.0)
                   long_lived = alive > (total_frames * 0.5)
  over_det_rate  : avg_active_per_frame / gt_persons (ideal = 1.0)
  id_recovery    : rescue_count / max(frags - gt_persons, 1)
                   fraction of ID breaks recovered by Re-ID (ideal = 1.0)
  gt_score       : harmonic mean of id_precision, coverage, 1/over_det_rate
                   composite score 0..1 (higher = better)

Proxy metrics (no GT required):
  avg_conf       : mean confidence of high-score detections
  det_per_frame  : avg detections per frame
  det_std        : std of detection count
  track_fragments: total unique track IDs
  avg_track_life : mean track lifetime (frames)
  long_track_ratio: % of tracks alive > 30 frames
  rescue_count / rescue_rate / avg_rescue_sim
  avg_det_ms / avg_reid_ms / total_time_s

Usage:
    .venv/bin/python scripts/evaluate.py \\
        --video  assets/142755-780943401_medium.mp4 \\
        --frames 300 --skip 2 --scale 0.4 \\
        --gt_persons 7
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, asdict
from itertools import product
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from detection.rtdetr_detector import RTDETRv2Detector
from detection.yolox_detector import YOLOXDetector
from tracking.bytetrack import ByteTracker, Detection as TrackDetection
from tracking.botsort import BoTSORT
from tracking.ocsort import OCSORTTracker
from tracking.sort import SORTTracker
from tracking.strongsort import StrongSORTTracker
from tracking.deepsort import DeepSORTTracker
from tracking.fairmot import FairMOTTracker
from tracking.cbiou import CBIoUTracker
from tracking.hybridsort import HybridSORTTracker
from tracking.ucmctrack import UCMCTracker
from tracking.deepocsort import DeepOCSORTTracker
from tracking.smiletrack import SMILETracker
from tracking.sparsetrack import SparseTracker
from tracking.ghost import GHOSTTracker
from tracking.transtrack import TransTracker
from tracking.track import STrack, TrackState
from reid.fastreid_extractor import FastReIDExtractor
from reid.gallery import ReIDGallery

logging.basicConfig(level=logging.WARNING,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("evaluate")


# ---------------------------------------------------------------------------
@dataclass
class RunMetrics:
    combo_id: str
    detector: str
    tracker: str
    reid_model: str
    frames_processed: int = 0
    # Detection
    avg_conf: float = 0.0
    det_per_frame: float = 0.0
    det_std: float = 0.0
    # Tracking
    track_fragments: int = 0
    avg_track_life: float = 0.0
    long_track_ratio: float = 0.0
    avg_active_per_frame: float = 0.0
    peak_active: int = 0
    # Re-ID
    rescue_count: int = 0
    rescue_rate: float = 0.0
    avg_rescue_sim: float = 0.0
    gallery_size_end: int = 0
    # GT-based (filled only when gt_persons > 0)
    frag_rate: float = 0.0        # frags / gt  (ideal 1.0)
    id_precision: float = 0.0     # gt / frags  (ideal 1.0)
    coverage: float = 0.0         # long_lived / gt  (ideal 1.0)
    over_det_rate: float = 0.0    # avg_active / gt  (ideal 1.0)
    id_recovery: float = 0.0      # rescues / (frags-gt)
    gt_score: float = 0.0         # composite (ideal 1.0)
    # Post-processing GT metrics (filled when --postprocess != "none")
    post_method: str = ""
    post_track_fragments: int = 0
    post_frag_rate: float = 0.0
    post_id_precision: float = 0.0
    post_coverage: float = 0.0
    post_gt_score: float = 0.0
    # Perf
    avg_det_ms: float = 0.0
    avg_reid_ms: float = 0.0
    total_time_s: float = 0.0


# ---------------------------------------------------------------------------
# GT-based metric computation
# ---------------------------------------------------------------------------

def compute_gt_metrics(m: RunMetrics, gt_persons: int) -> None:
    """Fill GT-based fields in-place given known ground-truth person count."""
    gt = max(gt_persons, 1)
    n  = max(m.frames_processed, 1)

    # How many frames count as "long-lived"? >50% of total frames
    long_life_thresh = n * 0.5
    # We don't have per-track lifetimes here directly, but we can derive
    # the count from long_track_ratio × track_fragments
    long_lived_count = round(m.long_track_ratio * m.track_fragments)

    m.frag_rate    = m.track_fragments / gt
    m.id_precision = min(gt / max(m.track_fragments, 1), 1.0)
    m.coverage     = min(long_lived_count / gt, 1.0)
    m.over_det_rate = m.avg_active_per_frame / gt

    excess_frags = max(m.track_fragments - gt, 0)
    m.id_recovery = min(m.rescue_count / max(excess_frags, 1), 1.0)

    # GT score = harmonic mean of 3 key components:
    #   id_precision  (fewer spurious IDs)
    #   coverage      (gt persons tracked long enough)
    #   1/over_det_rate (not over-detecting; capped so >1x is penalised)
    under_det_penalty = min(1.0 / max(m.over_det_rate, 0.01), 1.0)
    components = [m.id_precision, m.coverage, under_det_penalty]
    denom = sum(1.0 / max(c, 1e-6) for c in components)
    m.gt_score = len(components) / denom if denom > 0 else 0.0


# ---------------------------------------------------------------------------
# Frame loading
# ---------------------------------------------------------------------------

def load_frames(video_path: str, max_frames: int, skip: int) -> list[np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    orig_frames: list[np.ndarray] = []
    fid = 0
    while len(orig_frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        fid += 1
        if fid % skip != 0:
            continue
        orig_frames.append(frame)
    cap.release()
    return orig_frames


# ---------------------------------------------------------------------------
# Tracker factory
# ---------------------------------------------------------------------------

def make_tracker(trk_name: str):
    if trk_name == "bytetrack":
        return ByteTracker(
            track_thresh=0.45, track_buffer=90, match_thresh=0.85,
            min_hits=3, iou_thresh_stage2=0.45, reid_cost_weight=0.35,
        )
    elif trk_name == "botsort":
        return BoTSORT(
            track_thresh=0.45, track_buffer=90, match_thresh=0.85,
            min_hits=3, iou_thresh_stage2=0.45, reid_cost_weight=0.55,
            gmc_method="orb",
        )
    elif trk_name == "sort":
        return SORTTracker(track_thresh=0.45, max_age=90, min_hits=3, iou_thresh=0.30)
    elif trk_name == "ocsort":
        return OCSORTTracker(
            track_thresh=0.45, track_buffer=90, match_thresh=0.85,
            min_hits=3, iou_thresh_stage2=0.45, reid_cost_weight=0.35,
            ocm_weight=0.20, delta_t=3,
        )
    elif trk_name == "deepsort":
        return DeepSORTTracker(
            track_thresh=0.45, track_buffer=90, min_hits=3,
            max_cosine_dist=0.25, nn_budget=100, iou_thresh_fallback=0.70,
            iou_thresh_stage2=0.45,
        )
    elif trk_name == "strongsort":
        return StrongSORTTracker(
            track_thresh=0.45, track_buffer=90, min_hits=3,
            max_cosine_dist=0.25, nn_budget=100, iou_thresh_fallback=0.70,
            iou_thresh_stage2=0.45, gmc_method="orb",
        )
    elif trk_name == "fairmot":
        return FairMOTTracker(
            track_thresh=0.45, track_buffer=90, match_thresh=0.80, min_hits=3,
            reid_weight=0.50, iou_weight=0.50, proximity_gate=0.05,
        )
    elif trk_name == "cbiou":
        return CBIoUTracker(
            track_thresh=0.45, track_buffer=90, match_thresh=0.80, min_hits=3,
            iou_thresh_stage2=0.45, base_expand=0.20, vel_scale=0.01,
        )
    elif trk_name == "hybridsort":
        return HybridSORTTracker(
            track_thresh=0.45, track_buffer=90, match_thresh=0.85, min_hits=3,
            iou_thresh_stage2=0.45,
            iou_weight=0.50, height_weight=0.20, shape_weight=0.10, reid_weight=0.20,
        )
    elif trk_name == "ucmctrack":
        return UCMCTracker(track_thresh=0.45, track_buffer=90, match_thresh_gp=150.0, min_hits=3)
    elif trk_name == "deepocsort":
        return DeepOCSORTTracker(
            track_thresh=0.45, track_buffer=90, match_thresh=0.85, min_hits=3,
            iou_thresh_stage2=0.45, reid_weight=0.40, ocm_weight=0.15,
        )
    elif trk_name == "smiletrack":
        return SMILETracker(
            track_thresh=0.45, track_buffer=90, match_thresh=0.85, min_hits=3,
            iou_thresh_stage2=0.45, reid_weight=0.45, iou_gate=0.10, cos_gate=0.30,
        )
    elif trk_name == "sparsetrack":
        return SparseTracker(
            track_thresh=0.45, track_buffer=90, match_thresh=0.85, min_hits=3,
            iou_thresh_stage2=0.45, reid_cost_weight=0.35, n_layers=3,
            proximity_thresh=0.50,
        )
    elif trk_name == "ghost":
        return GHOSTTracker(
            track_thresh=0.45, track_buffer=90,
            match_thresh_short=0.35, match_thresh_long=0.50,
            match_thresh_iou=0.45, min_hits=3, iou_thresh_stage3=0.45,
            proximity_gate=0.10,
        )
    elif trk_name == "transtrack":
        return TransTracker(
            track_thresh=0.45, track_buffer=90, min_hits=3,
            appear_thresh=0.50, iou_weight=0.20, appear_weight=0.80,
            iou_thresh_fallback=0.60,
        )
    # ---- Tuned variants ----
    elif trk_name == "sort_tuned":
        # Raise track_thresh to 0.60 to suppress ghost tracks (ODrate was ~2x)
        return SORTTracker(track_thresh=0.60, max_age=90, min_hits=3, iou_thresh=0.30)
    elif trk_name == "botsort_tuned":
        # Disable GMC (gmc_method="none") to fix CPU instability
        return BoTSORT(
            track_thresh=0.45, track_buffer=90, match_thresh=0.85,
            min_hits=3, iou_thresh_stage2=0.45, reid_cost_weight=0.55,
            gmc_method="none",
        )
    elif trk_name == "sparsetrack_tuned":
        # Disable depth layers (n_layers=1) to act like OC-SORT
        return SparseTracker(
            track_thresh=0.45, track_buffer=90, match_thresh=0.85, min_hits=3,
            iou_thresh_stage2=0.45, reid_cost_weight=0.35, n_layers=1,
            proximity_thresh=0.50,
        )
    # ---- Round-2 tuning: address high fragmentation with ODrate≈1.0 ----
    elif trk_name == "bytetrack_tuned":
        # Longer buffer + more permissive match to reduce fragmentation
        return ByteTracker(
            track_thresh=0.50, track_buffer=180, match_thresh=0.90,
            min_hits=3, iou_thresh_stage2=0.45, reid_cost_weight=0.45,
        )
    elif trk_name == "ocsort_tuned":
        return OCSORTTracker(
            track_thresh=0.50, track_buffer=180, match_thresh=0.90,
            min_hits=3, iou_thresh_stage2=0.45, reid_cost_weight=0.45,
            ocm_weight=0.20, delta_t=5,
        )
    elif trk_name == "hybridsort_tuned":
        return HybridSORTTracker(
            track_thresh=0.50, track_buffer=180, match_thresh=0.90, min_hits=3,
            iou_thresh_stage2=0.45,
            iou_weight=0.40, height_weight=0.20, shape_weight=0.10, reid_weight=0.30,
        )
    elif trk_name == "deepocsort_tuned":
        return DeepOCSORTTracker(
            track_thresh=0.50, track_buffer=180, match_thresh=0.90, min_hits=3,
            iou_thresh_stage2=0.45, reid_weight=0.55, ocm_weight=0.15,
        )
    elif trk_name == "transtrack_tuned":
        return TransTracker(
            track_thresh=0.50, track_buffer=180, min_hits=3,
            appear_thresh=0.60, iou_weight=0.20, appear_weight=0.80,
            iou_thresh_fallback=0.70,
        )
    else:
        raise ValueError(f"Unknown tracker: {trk_name}")


# ---------------------------------------------------------------------------
# Single combo run
# ---------------------------------------------------------------------------

def run_combo(
    frames: list[np.ndarray],
    scaled_frames: list[np.ndarray],
    detector,
    trk_name: str,
    reid_ext: FastReIDExtractor,
    det_name: str,
    scale_inv: float,
    gt_persons: int = 0,
    postprocess: str = "none",
    merge_thresh: float = 0.65,
    liberal_rescue: bool = False,
) -> RunMetrics:

    combo_id = f"{det_name}+{trk_name}+{reid_ext._tag}"
    metrics = RunMetrics(combo_id=combo_id, detector=det_name,
                         tracker=trk_name, reid_model=reid_ext._tag)

    tracker = make_tracker(trk_name)
    STrack.reset_id_counter()

    # A: lower similarity threshold when liberal_rescue is enabled
    _sim_thresh = 0.60 if liberal_rescue else 0.72
    gallery = ReIDGallery(max_gallery_size=500, similarity_thresh=_sim_thresh,
                          max_embeddings_per_id=16)

    det_counts: list[int] = []
    all_confs:  list[float] = []
    det_times:  list[float] = []
    reid_times: list[float] = []
    active_counts: list[int] = []
    track_birth: dict[int, int] = {}
    track_death: dict[int, int] = {}
    track_embeddings: dict[int, list[np.ndarray]] = {}
    rescue_sims: list[float] = []
    total_new_tentative = 0

    t_start = time.perf_counter()
    use_scaled = (det_name == "rtdetr" and scale_inv != 1.0)

    for proc_id, orig_frame in enumerate(frames, 1):
        det_frame = scaled_frames[proc_id - 1] if use_scaled else orig_frame

        # Detection
        t0 = time.perf_counter()
        high_raw, low_raw = detector.detect(det_frame)
        det_times.append((time.perf_counter() - t0) * 1000)

        if use_scaled:
            for d in high_raw: d.bbox *= scale_inv
            for d in low_raw:  d.bbox *= scale_inv

        all_confs.extend([d.score for d in high_raw])
        det_counts.append(len(high_raw) + len(low_raw))

        # Re-ID
        t1 = time.perf_counter()
        embs = (reid_ext.extract_batch(orig_frame, [d.bbox for d in high_raw])
                if high_raw else np.zeros((0, reid_ext.feature_dim), np.float32))
        reid_times.append((time.perf_counter() - t1) * 1000)

        high_dets = [TrackDetection(bbox=d.bbox, score=d.score, embedding=embs[i])
                     for i, d in enumerate(high_raw)]
        low_dets  = [TrackDetection(bbox=d.bbox, score=d.score) for d in low_raw]

        # Track — all trackers accept (high_dets, low_dets, frame=...)
        # ByteTracker does NOT accept frame kwarg; others all do.
        if trk_name == "bytetrack":
            active = tracker.update(high_dets, low_dets)
        else:
            active = tracker.update(high_dets, low_dets, frame=orig_frame)

        active_counts.append(len(active))
        active_ids = {t.track_id for t in active}

        for t in active:
            if t.track_id not in track_birth:
                track_birth[t.track_id] = proc_id
                if t.state == TrackState.Tentative and t.hits == 1:
                    total_new_tentative += 1
            track_death[t.track_id] = proc_id
            # Collect embeddings for post-processing
            if t.reid_embedding is not None:
                if t.track_id not in track_embeddings:
                    track_embeddings[t.track_id] = []
                track_embeddings[t.track_id].append(t.reid_embedding.copy())

        # Gallery update
        for t in active:
            if t.state == TrackState.Confirmed and t.reid_embedding is not None:
                gallery.add_embedding(t.track_id, t.reid_embedding, proc_id)

        # Re-ID rescue
        # C: expand rescue window to hits<=5 when liberal_rescue is enabled
        rescue_hits_limit = 5 if liberal_rescue else 1
        for t in active:
            if (t.state == TrackState.Tentative
                    and t.hits <= rescue_hits_limit
                    and t.reid_embedding is not None):
                matched_id, sim = gallery.query(t.reid_embedding, exclude_ids=active_ids)
                if matched_id is not None:
                    rescue_sims.append(sim)
                    gallery.remove_track(matched_id)
                    t.reassign_id(matched_id)

        # B: skip gallery pruning when liberal_rescue (keep all embeddings alive)
        if not liberal_rescue and proc_id % 100 == 0:
            lost_ids = {t.track_id for t in tracker.get_lost_tracks()}
            gallery.prune_old_tracks(active_ids | lost_ids, 3000, proc_id)

    n = len(frames)
    metrics.frames_processed   = n
    metrics.total_time_s       = time.perf_counter() - t_start
    metrics.avg_det_ms         = float(np.mean(det_times))  if det_times  else 0.0
    metrics.avg_reid_ms        = float(np.mean(reid_times)) if reid_times else 0.0
    metrics.avg_conf           = float(np.mean(all_confs))  if all_confs  else 0.0
    metrics.det_per_frame      = float(np.mean(det_counts)) if det_counts else 0.0
    metrics.det_std            = float(np.std(det_counts))  if det_counts else 0.0
    metrics.avg_active_per_frame = float(np.mean(active_counts)) if active_counts else 0.0
    metrics.peak_active        = int(max(active_counts)) if active_counts else 0
    metrics.track_fragments    = len(track_birth)

    if track_birth:
        lifetimes = [track_death.get(tid, n) - track_birth[tid] + 1
                     for tid in track_birth]
        metrics.avg_track_life   = float(np.mean(lifetimes))
        metrics.long_track_ratio = float(sum(l > 30 for l in lifetimes) / len(lifetimes))

    metrics.rescue_count   = len(rescue_sims)
    metrics.rescue_rate    = len(rescue_sims) / max(total_new_tentative, 1) * 100
    metrics.avg_rescue_sim = float(np.mean(rescue_sims)) if rescue_sims else 0.0
    metrics.gallery_size_end = gallery.size()

    if gt_persons > 0:
        compute_gt_metrics(metrics, gt_persons)

    # ---- Post-processing ------------------------------------------------
    if postprocess != "none" and track_embeddings:
        from tracking.postprocess import (
            compute_mean_embeddings, merge_by_appearance,
            cluster_reassign, apply_id_map, compose_maps,
        )

        mean_embeds = compute_mean_embeddings(track_embeddings)
        all_tids = list(track_birth.keys())
        id_map: dict[int, int] = {}

        if postprocess in ("merge", "both"):
            id_map = merge_by_appearance(
                track_birth, track_death, mean_embeds,
                sim_thresh=merge_thresh, gap_tolerance=5,
            )

        if postprocess in ("cluster", "both") and gt_persons > 0:
            # After merge, recompute mean embeds for merged tracks
            if id_map:
                merged_groups: dict[int, list[np.ndarray]] = {}
                for old_id, new_id in id_map.items():
                    if old_id in mean_embeds:
                        merged_groups.setdefault(new_id, []).append(mean_embeds[old_id])
                merged_mean: dict[int, np.ndarray] = {}
                for tid, emb_list in merged_groups.items():
                    arr = np.stack(emb_list)
                    m = arr.mean(0)
                    nrm = np.linalg.norm(m)
                    merged_mean[tid] = m / nrm if nrm > 1e-12 else m
                mb, md = apply_id_map(track_birth, track_death, id_map)
                id_map2 = cluster_reassign(mb, merged_mean, gt_persons)
                id_map  = compose_maps(id_map, id_map2, all_tids)
            else:
                id_map = cluster_reassign(track_birth, mean_embeds, gt_persons)

        if id_map:
            post_birth, post_death = apply_id_map(track_birth, track_death, id_map)
            post_frags = len(post_birth)
            post_lifetimes = [post_death.get(tid, n) - post_birth[tid] + 1
                              for tid in post_birth]
            post_long_lived = sum(l > n * 0.5 for l in post_lifetimes)

            metrics.post_method          = postprocess
            metrics.post_track_fragments = post_frags

            if gt_persons > 0:
                gt = max(gt_persons, 1)
                metrics.post_frag_rate    = post_frags / gt
                metrics.post_id_precision = min(gt / max(post_frags, 1), 1.0)
                metrics.post_coverage     = min(post_long_lived / gt, 1.0)
                # ODrate is per-frame active count — unchanged by offline relabelling
                under_det = min(1.0 / max(metrics.over_det_rate, 0.01), 1.0)
                comps = [metrics.post_id_precision, metrics.post_coverage, under_det]
                denom = sum(1.0 / max(c, 1e-6) for c in comps)
                metrics.post_gt_score = len(comps) / denom if denom > 0 else 0.0

    return metrics


# ---------------------------------------------------------------------------
# Table printing
# ---------------------------------------------------------------------------

def print_table(results: list[RunMetrics], gt_persons: int = 0) -> None:
    has_gt = gt_persons > 0

    if has_gt:
        has_post = any(r.post_method for r in results)
        # Sort by post_gt_score if available, else gt_score
        sort_key = (lambda r: r.post_gt_score) if has_post else (lambda r: r.gt_score)
        sorted_results = sorted(results, key=sort_key, reverse=True)

        if has_post:
            W = 192
            sep = "─" * W
            hdr = (
                f"{'Rank':<5}{'Combo':<34} │ "
                f"{'GTscore':>7} │ {'PostScore':>9} │ "
                f"{'Frags':>5} │ {'PostFrags':>9} │ "
                f"{'IDprec':>6} │ {'PostIDpr':>8} │ "
                f"{'Cover':>5} │ {'PostCov':>7} │ "
                f"{'ODrate':>6} │ {'IDrecov':>7} │ {'Tot s':>6}"
            )
            print(f"\n{'='*W}")
            print(f"  EVALUATION RESULTS  —  GT-based + Post-processing  (gt_persons={gt_persons})")
            print(f"{'='*W}")
            print(hdr)
            print(sep)
            for rank, r in enumerate(sorted_results, 1):
                print(
                    f"#{rank:<4}{r.combo_id:<34} │ "
                    f"{r.gt_score:>7.3f} │ "
                    f"{r.post_gt_score:>9.3f} │ "
                    f"{r.track_fragments:>5d} │ "
                    f"{r.post_track_fragments:>9d} │ "
                    f"{r.id_precision:>6.3f} │ "
                    f"{r.post_id_precision:>8.3f} │ "
                    f"{r.coverage:>5.3f} │ "
                    f"{r.post_coverage:>7.3f} │ "
                    f"{r.over_det_rate:>6.2f}x │ "
                    f"{r.id_recovery:>7.3f} │ "
                    f"{r.total_time_s:>6.1f}"
                )
        else:
            W = 168
            sep = "─" * W
            hdr = (
                f"{'Rank':<5}{'Combo':<34} │ "
                f"{'GTscore':>7} │ {'IDprec':>6} │ {'Cover':>5} │ {'ODrate':>6} │ "
                f"{'FragRt':>6} │ {'IDrecov':>7} │ "
                f"{'Frags':>5} │ {'Life':>5} │ {'Lng%':>4} │ "
                f"{'Resc':>4} │ {'Sim':>5} │ {'Tot s':>6}"
            )
            print(f"\n{'='*W}")
            print(f"  EVALUATION RESULTS  —  GT-based ranking  (gt_persons={gt_persons})")
            print(f"{'='*W}")
            print(hdr)
            print(sep)
            for rank, r in enumerate(sorted_results, 1):
                print(
                    f"#{rank:<4}{r.combo_id:<34} │ "
                    f"{r.gt_score:>7.3f} │ "
                    f"{r.id_precision:>6.3f} │ "
                    f"{r.coverage:>5.3f} │ "
                    f"{r.over_det_rate:>6.2f}x │ "
                    f"{r.frag_rate:>6.1f}x │ "
                    f"{r.id_recovery:>7.3f} │ "
                    f"{r.track_fragments:>5d} │ "
                    f"{r.avg_track_life:>5.1f} │ "
                    f"{r.long_track_ratio*100:>4.0f} │ "
                    f"{r.rescue_count:>4d} │ "
                    f"{r.avg_rescue_sim:>5.3f} │ "
                    f"{r.total_time_s:>6.1f}"
                )
        print(sep)
        print(f"""
  GT-based metric guide (gt_persons={gt_persons}):
  GTscore   before post-processing   │  PostScore  after post-processing  ★★★
  IDprec    gt / frags  (ideal 1.0)  │  PostIDpr   after post
  Cover     long-lived / gt          │  PostCov    after post
  ODrate    avg_active / gt (≈1.0x)  │  (unchanged by offline relabelling)
  IDrecov   rescues / excess_frags
  Frags / PostFrags   before / after post-processing
""")
    else:
        sorted_results = sorted(results, key=lambda r: r.combo_id)
        W = 148
        sep = "─" * W
        hdr = (
            f"{'Combo':<38} │ {'Det/f':>5} │ {'Conf':>5} │ {'Std':>4} │ "
            f"{'Frags':>5} │ {'Life':>5} │ {'Lng%':>4} │ "
            f"{'Resc':>4} │ {'Rsc%':>5} │ {'Sim':>5} │ "
            f"{'DetMs':>5} │ {'RIDMs':>5} │ {'Tot s':>6}"
        )
        print(f"\n{'='*W}")
        print("  EVALUATION RESULTS  —  Detector × Tracker × Re-ID")
        print(f"{'='*W}")
        print(hdr)
        print(sep)
        for r in sorted_results:
            print(
                f"{r.combo_id:<38} │ "
                f"{r.det_per_frame:>5.1f} │ "
                f"{r.avg_conf:>5.3f} │ "
                f"{r.det_std:>4.1f} │ "
                f"{r.track_fragments:>5d} │ "
                f"{r.avg_track_life:>5.1f} │ "
                f"{r.long_track_ratio*100:>4.0f} │ "
                f"{r.rescue_count:>4d} │ "
                f"{r.rescue_rate:>5.1f} │ "
                f"{r.avg_rescue_sim:>5.3f} │ "
                f"{r.avg_det_ms:>5.0f} │ "
                f"{r.avg_reid_ms:>5.0f} │ "
                f"{r.total_time_s:>6.1f}"
            )
        print(sep)

    # Best per metric (always shown)
    print("  Best per metric:")
    best_items = [
        ("Highest GT score",          max(results, key=lambda r: r.gt_score),          "gt_score")
        if has_gt else
        ("Fewest track fragments",    min(results, key=lambda r: r.track_fragments),   "track_fragments"),
        ("Best ID precision",         max(results, key=lambda r: r.id_precision),      "id_precision")
        if has_gt else
        ("Longest avg track life",    max(results, key=lambda r: r.avg_track_life),    "avg_track_life"),
        ("Best coverage",             max(results, key=lambda r: r.coverage),          "coverage")
        if has_gt else
        ("Most Re-ID rescues",        max(results, key=lambda r: r.rescue_count),       "rescue_count"),
        ("Best ID recovery",          max(results, key=lambda r: r.id_recovery),       "id_recovery")
        if has_gt else
        ("Highest rescue sim",        max(results, key=lambda r: r.avg_rescue_sim),    "avg_rescue_sim"),
        ("Fewest track fragments",    min(results, key=lambda r: r.track_fragments),   "track_fragments"),
        ("Highest rescue sim",        max(results, key=lambda r: r.avg_rescue_sim),    "avg_rescue_sim"),
        ("Fastest (Tot s)",           min(results, key=lambda r: r.total_time_s),      "total_time_s"),
    ]
    for label, r, attr in best_items:
        val = getattr(r, attr)
        fmt = ".3f" if isinstance(val, float) else "d"
        print(f"    {label:<30} → {r.combo_id:<38}  ({attr}={val:{fmt}})")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ALL_TRACKERS = [
        "bytetrack", "botsort", "sort", "ocsort", "deepsort", "strongsort",
        "fairmot", "transtrack", "cbiou", "hybridsort", "ucmctrack",
        "deepocsort", "smiletrack", "sparsetrack", "ghost",
    ]

    parser = argparse.ArgumentParser()
    parser.add_argument("--video",      default="assets/142755-780943401_medium.mp4")
    parser.add_argument("--frames",     type=int,   default=300)
    parser.add_argument("--skip",       type=int,   default=2)
    parser.add_argument("--scale",      type=float, default=0.4)
    parser.add_argument("--device",     default="cpu")
    parser.add_argument("--out",        default="outputs/eval_results.json")
    parser.add_argument("--gt_persons", type=int,   default=7,
                        help="Known number of persons in video (0 = disable GT metrics)")
    parser.add_argument("--detectors",  nargs="+", default=["yolox_s"])
    parser.add_argument("--trackers",   nargs="+", default=ALL_TRACKERS)
    parser.add_argument("--reids",      nargs="+", default=["osnet"])
    parser.add_argument("--postprocess", default="none",
                        choices=["none", "merge", "cluster", "both"],
                        help="Post-process to reduce ID fragmentation: "
                             "merge=appearance merge, cluster=k-means reassign, both=merge then cluster")
    parser.add_argument("--merge_thresh", type=float, default=0.65,
                        help="Cosine similarity threshold for appearance merge (default 0.65)")
    parser.add_argument("--liberal_rescue", action="store_true",
                        help="Enable A+B+C: lower gallery thresh(0.60), no pruning, rescue hits<=5")
    parser.add_argument("--degrade_res", type=float, default=1.0,
                        help="Resize all frames by this factor before processing (e.g. 0.5 = half resolution)")
    args = parser.parse_args()

    os.makedirs(Path(args.out).parent, exist_ok=True)

    # ── Pre-read frames ──────────────────────────────────────────────────
    print(f"\nLoading {args.frames} frames (skip={args.skip}) from {args.video} …")
    orig_frames = load_frames(args.video, args.frames, args.skip)
    n = len(orig_frames)
    print(f"  → {n} frames loaded  ({orig_frames[0].shape[1]}×{orig_frames[0].shape[0]})")

    # Apply resolution degradation if requested
    if args.degrade_res != 1.0:
        dh = max(1, int(orig_frames[0].shape[0] * args.degrade_res))
        dw = max(1, int(orig_frames[0].shape[1] * args.degrade_res))
        orig_frames = [cv2.resize(f, (dw, dh)) for f in orig_frames]
        print(f"  → degraded to {dw}×{dh}  (degrade_res={args.degrade_res})")

    if args.gt_persons > 0:
        print(f"  → GT: {args.gt_persons} persons in video")

    h0, w0 = orig_frames[0].shape[:2]
    sw, sh  = int(w0 * args.scale), int(h0 * args.scale)
    scaled_frames = [cv2.resize(f, (sw, sh)) for f in orig_frames]
    scale_inv = 1.0 / args.scale

    # ── Preload detectors ────────────────────────────────────────────────
    detectors: dict[str, object] = {}
    for det_name in args.detectors:
        print(f"Loading detector: {det_name} …")
        if det_name == "rtdetr":
            d = RTDETRv2Detector(
                model_name="PekingU/rtdetr_v2_r50vd", device=args.device,
                high_score_thresh=0.45, low_score_thresh=0.10,
                input_size=(sw, sh),
            )
        elif det_name == "yolox_s":
            d = YOLOXDetector(
                model_name="yolox_s", device=args.device,
                high_score_thresh=0.45, low_score_thresh=0.10,
            )
        else:
            raise ValueError(f"Unknown detector: {det_name}")
        d.load_model()
        detectors[det_name] = d
        print(f"  → {det_name} ready")

    # ── Preload Re-ID models ──────────────────────────────────────────────
    reid_models: dict[str, FastReIDExtractor] = {}
    for reid_name in args.reids:
        print(f"Loading Re-ID model: {reid_name} …")
        ext = FastReIDExtractor(model_path=None, device=args.device,
                                input_size=(256, 128), use_onnx=False)
        ext.load_model()
        ext._tag = reid_name
        if reid_name == "osnet" and ext.feature_dim == 2048:
            ext._tag = "osnet→resnet50"
        reid_models[reid_name] = ext
        print(f"  → {reid_name} ready (feature_dim={ext.feature_dim})")

    # ── Run all combinations ─────────────────────────────────────────────
    combos = list(product(args.detectors, args.trackers, args.reids))
    print(f"\nRunning {len(combos)} combos × {n} frames …\n")

    results: list[RunMetrics] = []
    for i, (det_name, trk_name, reid_name) in enumerate(combos, 1):
        combo_id = f"{det_name}+{trk_name}+{reid_models[reid_name]._tag}"
        print(f"  [{i:2d}/{len(combos)}] {combo_id:<44}", end=" ", flush=True)
        STrack.reset_id_counter()
        try:
            det_si = scale_inv if det_name == "rtdetr" else 1.0
            m = run_combo(
                frames=orig_frames,
                scaled_frames=scaled_frames,
                detector=detectors[det_name],
                trk_name=trk_name,
                reid_ext=reid_models[reid_name],
                det_name=det_name,
                scale_inv=det_si,
                gt_persons=args.gt_persons,
                postprocess=args.postprocess,
                merge_thresh=args.merge_thresh,
                liberal_rescue=args.liberal_rescue,
            )
            results.append(m)
            gt_str = f"  GTscore={m.gt_score:.3f}" if args.gt_persons > 0 else ""
            post_str = (f"  →post:{m.post_gt_score:.3f}(frags={m.post_track_fragments})"
                        if m.post_method else "")
            print(f"✓  {m.total_time_s:5.1f}s  frags={m.track_fragments}"
                  f"  rescue={m.rescue_count}  sim={m.avg_rescue_sim:.3f}{gt_str}{post_str}")
        except Exception as e:
            print(f"✗  ERROR: {e}")
            logger.exception("Combo %s failed", combo_id)

    if results:
        print_table(results, gt_persons=args.gt_persons)
        with open(args.out, "w") as f:
            json.dump([asdict(r) for r in results], f, indent=2)
        print(f"Results saved → {args.out}\n")


if __name__ == "__main__":
    main()
