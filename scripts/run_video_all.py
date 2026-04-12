"""
Exhaustive multi-tracker × post-processing comparison for a single video.

Generates one annotated video per (tracker × postproc) combination:
    {out_dir}/{stem}_{tracker}_{postproc}.mp4

Post-processing options per tracker:
    none   — raw tracking output
    abc    — online: lower gallery thresh (A) + no pruning (B) + wider rescue (C)
    de     — offline: appearance-merge (D) then cluster-reassign (E)
    abcde  — online ABC + offline DE

Trackers (23 total):
    Base (15): bytetrack, botsort, sort, ocsort, deepsort, strongsort,
               fairmot, cbiou, hybridsort, ucmctrack, deepocsort,
               smiletrack, sparsetrack, ghost, transtrack
    Tuned (8): sort_tuned, botsort_tuned, sparsetrack_tuned,
               bytetrack_tuned, ocsort_tuned, hybridsort_tuned,
               deepocsort_tuned, transtrack_tuned

Usage:
    .venv/bin/python scripts/run_video_all.py \
        --video assets/test2.mp4 \
        --out_dir outputs/all_combos \
        --gt_persons 7 \
        --skip 2
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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
from tracking.postprocess import (
    compute_mean_embeddings, merge_by_appearance,
    cluster_reassign, apply_id_map, compose_maps,
)

logging.basicConfig(level=logging.WARNING,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("run_video_all")

# ── tracker colour palette ─────────────────────────────────────────────────────
TRACKER_COLORS: dict[str, tuple[int, int, int]] = {
    "sort":              (150, 150,  30),
    "sort_tuned":        (  0, 200, 255),
    "bytetrack":         (  0, 120, 255),
    "bytetrack_tuned":   ( 80, 160, 255),
    "ocsort":            (  0, 200, 100),
    "ocsort_tuned":      ( 50, 230,  50),
    "hybridsort":        (200, 120,   0),
    "hybridsort_tuned":  (255, 165,   0),
    "deepocsort":        (160,   0, 160),
    "deepocsort_tuned":  (220,   0, 220),
    "botsort":           (200,  80,  20),
    "botsort_tuned":     (255, 120,  40),
    "deepsort":          ( 20, 180, 180),
    "strongsort":        (  0, 140, 200),
    "fairmot":           (120, 200,   0),
    "cbiou":             (200, 200,   0),
    "hybridsort":        (200, 140,   0),
    "ucmctrack":         (100, 100, 200),
    "smiletrack":        (  0, 220, 180),
    "sparsetrack":       (180, 100, 200),
    "sparsetrack_tuned": (210, 130, 230),
    "ghost":             (140, 140, 140),
    "transtrack":        (100, 200, 200),
    "transtrack_tuned":  (140, 230, 230),
}
DEFAULT_COLOR = (180, 180, 180)

# post-processing banner accent colours
POSTPROC_ACCENT: dict[str, tuple[int, int, int]] = {
    "none":  (30,  30,  30),
    "abc":   (20, 130,  20),
    "de":    (180,  30,  30),
    "abcde": (120,  20, 180),
}
POSTPROC_LABEL: dict[str, str] = {
    "none":  "",
    "abc":   "+ABC",
    "de":    "+DE",
    "abcde": "+ABCDE",
}

_STATE_COLORS = {
    TrackState.Tentative: (200, 200,   0),
    TrackState.Confirmed: (  0, 230,   0),
    TrackState.Lost:      (  0, 120, 255),
}

ALL_TRACKERS = [
    # base
    "bytetrack", "botsort", "sort", "ocsort", "deepsort", "strongsort",
    "fairmot", "cbiou", "hybridsort", "ucmctrack", "deepocsort",
    "smiletrack", "sparsetrack", "ghost", "transtrack",
    # tuned
    "sort_tuned", "botsort_tuned", "sparsetrack_tuned",
    "bytetrack_tuned", "ocsort_tuned", "hybridsort_tuned",
    "deepocsort_tuned", "transtrack_tuned",
]
ALL_POSTPROC = ["none", "abc", "de", "abcde"]


# ── tracker factory ────────────────────────────────────────────────────────────

def make_tracker(name: str):
    if name == "bytetrack":
        return ByteTracker(track_thresh=0.45, track_buffer=90, match_thresh=0.85,
                           min_hits=3, iou_thresh_stage2=0.45, reid_cost_weight=0.35)
    if name == "botsort":
        return BoTSORT(track_thresh=0.45, track_buffer=90, match_thresh=0.85,
                       min_hits=3, iou_thresh_stage2=0.45, gmc_method="orb")
    if name == "sort":
        return SORTTracker(track_thresh=0.45, max_age=90, min_hits=3, iou_thresh=0.30)
    if name == "ocsort":
        return OCSORTTracker(track_thresh=0.45, track_buffer=90, match_thresh=0.85,
                             min_hits=3, iou_thresh_stage2=0.45, reid_cost_weight=0.35,
                             ocm_weight=0.20, delta_t=3)
    if name == "deepsort":
        return DeepSORTTracker(track_thresh=0.45, track_buffer=90, min_hits=3,
                               max_cosine_dist=0.25, iou_thresh_stage2=0.45)
    if name == "strongsort":
        return StrongSORTTracker(track_thresh=0.45, track_buffer=90, min_hits=3,
                                 max_cosine_dist=0.25, iou_thresh_stage2=0.45,
                                 gmc_method="orb")
    if name == "fairmot":
        return FairMOTTracker(track_thresh=0.45, track_buffer=90, match_thresh=0.80,
                              min_hits=3, reid_weight=0.50, iou_weight=0.50,
                              proximity_gate=0.05)
    if name == "cbiou":
        return CBIoUTracker(track_thresh=0.45, track_buffer=90, match_thresh=0.80,
                            min_hits=3, iou_thresh_stage2=0.45,
                            base_expand=0.20, vel_scale=0.01)
    if name == "hybridsort":
        return HybridSORTTracker(track_thresh=0.45, track_buffer=90, match_thresh=0.85,
                                 min_hits=3, iou_thresh_stage2=0.45,
                                 iou_weight=0.50, height_weight=0.20,
                                 shape_weight=0.10, reid_weight=0.20)
    if name == "ucmctrack":
        return UCMCTracker(track_thresh=0.45, track_buffer=90,
                           match_thresh_gp=150.0, min_hits=3)
    if name == "deepocsort":
        return DeepOCSORTTracker(track_thresh=0.45, track_buffer=90, match_thresh=0.85,
                                 min_hits=3, iou_thresh_stage2=0.45,
                                 reid_weight=0.40, ocm_weight=0.15)
    if name == "smiletrack":
        return SMILETracker(track_thresh=0.45, track_buffer=90, match_thresh=0.85,
                            min_hits=3, iou_thresh_stage2=0.45,
                            reid_weight=0.45, iou_gate=0.10, cos_gate=0.30)
    if name == "sparsetrack":
        return SparseTracker(track_thresh=0.45, track_buffer=90, match_thresh=0.85,
                             min_hits=3, iou_thresh_stage2=0.45, proximity_thresh=0.50)
    if name == "ghost":
        return GHOSTTracker(track_thresh=0.45, track_buffer=90,
                            match_thresh_short=0.35, match_thresh_long=0.50,
                            match_thresh_iou=0.45, min_hits=3,
                            proximity_gate=0.10)
    if name == "transtrack":
        return TransTracker(track_thresh=0.45, track_buffer=90, min_hits=3,
                            appear_thresh=0.65, iou_weight=0.30,
                            appear_weight=0.70, iou_thresh_fallback=0.70)
    # ── tuned variants ────────────────────────────────────────────────────────
    if name == "sort_tuned":
        return SORTTracker(track_thresh=0.60, max_age=90, min_hits=3, iou_thresh=0.30)
    if name == "botsort_tuned":
        return BoTSORT(track_thresh=0.45, track_buffer=90, match_thresh=0.85,
                       min_hits=3, iou_thresh_stage2=0.45, gmc_method="none")
    if name == "sparsetrack_tuned":
        return SparseTracker(track_thresh=0.45, track_buffer=90, match_thresh=0.85,
                             min_hits=3, iou_thresh_stage2=0.45,
                             proximity_thresh=0.50, n_layers=1)
    if name == "bytetrack_tuned":
        return ByteTracker(track_thresh=0.50, track_buffer=180, match_thresh=0.90,
                           min_hits=3, iou_thresh_stage2=0.45, reid_cost_weight=0.45)
    if name == "ocsort_tuned":
        return OCSORTTracker(track_thresh=0.50, track_buffer=180, match_thresh=0.90,
                             min_hits=3, iou_thresh_stage2=0.45, reid_cost_weight=0.45,
                             ocm_weight=0.20, delta_t=5)
    if name == "hybridsort_tuned":
        return HybridSORTTracker(track_thresh=0.50, track_buffer=180, match_thresh=0.90,
                                 min_hits=3, iou_thresh_stage2=0.45,
                                 iou_weight=0.40, height_weight=0.20,
                                 shape_weight=0.10, reid_weight=0.30)
    if name == "deepocsort_tuned":
        return DeepOCSORTTracker(track_thresh=0.50, track_buffer=180, match_thresh=0.90,
                                 min_hits=3, iou_thresh_stage2=0.45,
                                 reid_weight=0.55, ocm_weight=0.15)
    if name == "transtrack_tuned":
        return TransTracker(track_thresh=0.50, track_buffer=180, min_hits=3,
                            appear_thresh=0.60, iou_weight=0.20,
                            appear_weight=0.80, iou_thresh_fallback=0.70)
    raise ValueError(f"Unknown tracker: {name}")


# ── per-frame track record ─────────────────────────────────────────────────────

class FT(NamedTuple):
    """Lightweight per-frame track snapshot."""
    track_id: int
    tlbr: np.ndarray
    state: TrackState


# ── drawing ────────────────────────────────────────────────────────────────────

def draw_frame(
    frame: np.ndarray,
    ft_list: list[FT],
    id_map: dict[int, int],
    tracker_name: str,
    postproc: str,
    frame_id: int,
    fps: float,
    inference_ms: float,
    frag_count: int,
    rescue_count: int,
    reid_model: str = "osnet_x0_25",
) -> np.ndarray:
    frame_h, frame_w = frame.shape[:2]
    out = frame.copy()

    tag_color    = TRACKER_COLORS.get(tracker_name, DEFAULT_COLOR)
    accent_color = POSTPROC_ACCENT[postproc]
    pp_label     = POSTPROC_LABEL[postproc]

    # ── bounding boxes ────────────────────────────────────────────────────────
    for ft in ft_list:
        disp_id = id_map.get(ft.track_id, ft.track_id)
        x1, y1, x2, y2 = ft.tlbr.astype(int)
        state_color = _STATE_COLORS.get(ft.state, (180, 180, 180))

        cv2.rectangle(out, (x1, y1), (x2, y2), state_color, 3)

        box_h      = max(y2 - y1, 1)
        font_scale = max(0.7, min(2.0, box_h / 220))
        thickness  = max(1, int(font_scale * 1.5))
        label      = f" ID {disp_id} "
        font       = cv2.FONT_HERSHEY_DUPLEX
        (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        by2 = max(th + baseline + 8, y1)
        by1 = max(0, y1 - th - baseline - 8)
        cv2.rectangle(out, (x1, by1), (x1 + tw, by2), state_color, cv2.FILLED)
        cv2.putText(out, label, (x1, by2 - baseline - 3),
                    font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)

    # ── tracker name banner (top-centre) ──────────────────────────────────────
    sf       = frame_h / 720
    tf       = cv2.FONT_HERSHEY_DUPLEX
    t_fs     = 1.5 * sf
    t_th     = max(2, int(sf * 2.5))
    tag_text = f"  {tracker_name.upper()}{pp_label}  "
    (tw, th), tbl = cv2.getTextSize(tag_text, tf, t_fs, t_th)
    pad_x = int(20 * sf);  pad_y = int(10 * sf)
    bx1 = (frame_w - tw) // 2 - pad_x;  by1 = int(10 * sf)
    bx2 = (frame_w + tw) // 2 + pad_x;  by2 = by1 + th + tbl + pad_y * 2

    # coloured main background
    ovl = out.copy()
    cv2.rectangle(ovl, (bx1, by1), (bx2, by2), tag_color, cv2.FILLED)
    cv2.addWeighted(ovl, 0.85, out, 0.15, 0, out)
    cv2.putText(out, tag_text,
                ((frame_w - tw) // 2, by1 + pad_y + th),
                tf, t_fs, (0, 0, 0), t_th, cv2.LINE_AA)
    # accent border for post-processing visibility
    if postproc != "none":
        cv2.rectangle(out, (bx1, by1), (bx2, by2), accent_color, max(3, int(4 * sf)))

    # ── stats panel (top-left) ────────────────────────────────────────────────
    sf2 = sf
    sf_font = cv2.FONT_HERSHEY_DUPLEX
    sf_fs   = 0.60 * sf2;  sf_th = max(1, int(sf2));  lh = int(30 * sf2);  pd = int(14 * sf2)
    stats = [
        f"Tracker : {tracker_name}",
        f"Post    : {postproc.upper()}",
        f"ReID    : {reid_model}",
        f"Frame   : {frame_id}",
        f"FPS     : {fps:.1f}",
        f"Infer   : {inference_ms:.0f} ms",
        f"Active  : {sum(1 for ft in ft_list if ft.state == TrackState.Confirmed)}",
        f"Frags   : {frag_count}",
        f"Rescue  : {rescue_count}",
    ]
    (pw, _), _ = cv2.getTextSize("Tracker : sort_tuned", sf_font, sf_fs, sf_th)
    pw2 = pw + pd * 2;  ph = lh * len(stats) + pd * 2
    ovl2 = out.copy()
    cv2.rectangle(ovl2, (0, 0), (pw2, ph), (15, 15, 15), cv2.FILLED)
    cv2.addWeighted(ovl2, 0.60, out, 0.40, 0, out)
    for i, text in enumerate(stats):
        cv2.putText(out, text, (pd, pd + lh * (i + 1) - int(5 * sf2)),
                    sf_font, sf_fs, (220, 220, 220), sf_th, cv2.LINE_AA)

    return out


# ── one tracking pass ──────────────────────────────────────────────────────────

def tracking_pass(
    frames: list[np.ndarray],
    tracker_name: str,
    detector: YOLOXDetector,
    reid_ext: FastReIDExtractor,
    liberal: bool = False,
) -> tuple[
    list[list[FT]],          # frame_data
    dict[int, list[np.ndarray]],  # track_embeddings
    dict[int, int],           # track_birth
    dict[int, int],           # track_death
    dict,                     # raw metrics
]:
    STrack.reset_id_counter()
    tracker = make_tracker(tracker_name)

    # Gallery settings
    if liberal:
        gallery = ReIDGallery(max_gallery_size=500, similarity_thresh=0.60,
                              max_embeddings_per_id=16)
    else:
        gallery = ReIDGallery(max_gallery_size=200, similarity_thresh=0.72,
                              max_embeddings_per_id=8)

    frame_data:       list[list[FT]]              = []
    track_embeddings: dict[int, list[np.ndarray]] = {}
    track_birth:      dict[int, int]              = {}
    track_death:      dict[int, int]              = {}
    rescue_count  = 0
    rescue_sims:  list[float] = []
    det_times:    list[float] = []
    active_cnts:  list[int]   = []

    for fid, frame in enumerate(frames, 1):
        t0 = time.perf_counter()
        high_raw, low_raw = detector.detect(frame)
        det_times.append((time.perf_counter() - t0) * 1000)

        high_embs = reid_ext.extract_batch(frame, [d.bbox for d in high_raw]) \
            if high_raw else np.zeros((0, reid_ext.feature_dim), np.float32)

        high_dets = [TrackDetection(bbox=d.bbox, score=d.score, embedding=high_embs[i])
                     for i, d in enumerate(high_raw)]
        low_dets  = [TrackDetection(bbox=d.bbox, score=d.score) for d in low_raw]

        active = tracker.update(high_dets, low_dets)
        active_ids = {t.track_id for t in active}

        confirmed = [t for t in active if t.state == TrackState.Confirmed]
        for t in confirmed:
            if t.reid_embedding is not None:
                gallery.add_embedding(t.track_id, t.reid_embedding, fid)
                track_embeddings.setdefault(t.track_id, []).append(t.reid_embedding)
            track_birth.setdefault(t.track_id, fid)
            track_death[t.track_id] = fid

        # Re-ID rescue
        rescue_hits_limit = 5 if liberal else 1
        for t in active:
            if (t.state == TrackState.Tentative
                    and t.hits <= rescue_hits_limit
                    and t.reid_embedding is not None):
                mid, sim = gallery.query(t.reid_embedding, exclude_ids=active_ids)
                if mid is not None:
                    rescue_count += 1
                    rescue_sims.append(sim)
                    old_id = t.track_id
                    t.reassign_id(mid)
                    gallery.remove_track(old_id)

        if fid % 150 == 0 and not liberal:
            lost = {t.track_id for t in tracker.get_lost_tracks()}
            gallery.prune_old_tracks(active_ids | lost, 3000, fid)

        active_cnts.append(len(active))
        frame_data.append([FT(t.track_id, t.tlbr.copy(), t.state) for t in active])

    raw = {
        "rescue_count":  rescue_count,
        "rescue_sims":   rescue_sims,
        "avg_det_ms":    float(np.mean(det_times)) if det_times else 0.0,
        "avg_active":    float(np.mean(active_cnts)) if active_cnts else 0.0,
        "all_tids":      set(track_birth.keys()),
    }
    return frame_data, track_embeddings, track_birth, track_death, raw


# ── compute metrics dict ───────────────────────────────────────────────────────

def compute_metrics(
    tracker_name: str,
    postproc: str,
    frame_data: list[list[FT]],
    id_map: dict[int, int],
    track_birth: dict[int, int],
    track_death: dict[int, int],
    raw: dict,
    gt_persons: int,
    total_s: float,
) -> dict:
    n_frames = len(frame_data)

    # Apply id_map to get effective unique IDs
    eff_ids: set[int] = set()
    for fts in frame_data:
        for ft in fts:
            eff_ids.add(id_map.get(ft.track_id, ft.track_id))
    frags = len(eff_ids)

    # Long-lived: compute from effective birth/death after id_map
    new_birth: dict[int, int] = {}
    new_death: dict[int, int] = {}
    for tid in track_birth:
        eid = id_map.get(tid, tid)
        new_birth[eid] = min(new_birth.get(eid, track_birth[tid]), track_birth[tid])
        new_death[eid] = max(new_death.get(eid, track_death.get(tid, 0)), track_death.get(tid, 0))
    long_lived = sum(1 for eid in eff_ids
                     if (new_death.get(eid, 0) - new_birth.get(eid, 0)) >= n_frames * 0.5)

    m: dict = {
        "tracker":       tracker_name,
        "postproc":      postproc,
        "frames":        n_frames,
        "frags":         frags,
        "rescue_count":  raw["rescue_count"],
        "avg_rescue_sim": float(np.mean(raw["rescue_sims"])) if raw["rescue_sims"] else 0.0,
        "avg_active":    raw["avg_active"],
        "avg_det_ms":    raw["avg_det_ms"],
        "total_s":       total_s,
    }
    if gt_persons > 0:
        gt = gt_persons
        id_prec  = min(gt / max(frags, 1), 1.0)
        coverage = min(long_lived / gt, 1.0)
        od_rate  = raw["avg_active"] / gt
        under_p  = min(1.0 / max(od_rate, 0.01), 1.0)
        comps    = [id_prec, coverage, under_p]
        denom    = sum(1.0 / max(c, 1e-6) for c in comps)
        gt_score = len(comps) / denom if denom > 0 else 0.0
        m.update({
            "gt_persons":   gt_persons,
            "id_precision": id_prec,
            "coverage":     coverage,
            "od_rate":      od_rate,
            "gt_score":     gt_score,
        })
    return m


# ── render one video ───────────────────────────────────────────────────────────

def render_video(
    frames: list[np.ndarray],
    frame_data: list[list[FT]],
    id_map: dict[int, int],
    tracker_name: str,
    postproc: str,
    out_path: Path,
    src_fps: float,
    skip: int,
    frag_count: int,
    rescue_count: int,
    inference_ms_per_frame: list[float],
    reid_model: str = "osnet_x0_25",
) -> None:
    frame_h, frame_w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, src_fps / skip, (frame_w, frame_h))

    fps_t0 = time.perf_counter();  fps_cnt = 0;  fps_disp = 0.0

    for fid, (frame, fts) in enumerate(zip(frames, frame_data), 1):
        fps_cnt += 1
        if fps_cnt >= 30:
            fps_disp = fps_cnt / (time.perf_counter() - fps_t0)
            fps_cnt  = 0;  fps_t0 = time.perf_counter()

        infer_ms = inference_ms_per_frame[fid - 1] if fid <= len(inference_ms_per_frame) else 0.0
        vis = draw_frame(frame, fts, id_map, tracker_name, postproc,
                         fid, fps_disp, infer_ms, frag_count, rescue_count,
                         reid_model=reid_model)
        writer.write(vis)

    writer.release()


# ── build DE id_map ────────────────────────────────────────────────────────────

def build_de_idmap(
    track_birth: dict[int, int],
    track_death: dict[int, int],
    track_embeddings: dict[int, list[np.ndarray]],
    gt_persons: int,
) -> dict[int, int]:
    all_tids = list(track_birth.keys())
    if not all_tids:
        return {}
    mean_embeds = compute_mean_embeddings(track_embeddings)
    # Strategy D
    id_map_d = merge_by_appearance(track_birth, track_death, mean_embeds,
                                   sim_thresh=0.65, gap_tolerance=5)
    # Strategy E (only if gt_persons known)
    if gt_persons > 0:
        # Compute merged birth/death for E
        new_b, new_d = apply_id_map(track_birth, track_death, id_map_d)
        # Recompute embeddings for merged IDs
        merged_embs: dict[int, list[np.ndarray]] = {}
        for old_id, emb_list in track_embeddings.items():
            new_id = id_map_d.get(old_id, old_id)
            merged_embs.setdefault(new_id, []).extend(emb_list)
        merged_mean = compute_mean_embeddings(merged_embs)
        try:
            id_map_e = cluster_reassign(new_b, merged_mean, gt_persons)
            return compose_maps(id_map_d, id_map_e, all_tids)
        except Exception:
            return id_map_d
    return id_map_d


# ── process one tracker (2 passes × 4 videos) ─────────────────────────────────

def process_tracker(
    frames: list[np.ndarray],
    tracker_name: str,
    detector: YOLOXDetector,
    reid_ext: FastReIDExtractor,
    out_dir: Path,
    stem: str,
    src_fps: float,
    skip: int,
    gt_persons: int,
    postproc_list: list[str],
    reid_model: str = "osnet_x0_25",
) -> list[dict]:
    results: list[dict] = []
    identity_map: dict[int, int] = {}  # placeholder, filled per pass

    # ── Pass 1: standard → none + DE ─────────────────────────────────────────
    need_std = any(p in postproc_list for p in ("none", "de"))
    if need_std:
        t_start = time.perf_counter()
        fd_std, embs_std, birth_std, death_std, raw_std = tracking_pass(
            frames, tracker_name, detector, reid_ext, liberal=False)
        pass1_s = time.perf_counter() - t_start

        det_times_std = [0.0] * len(frames)  # timing already merged into raw

        if "none" in postproc_list:
            id_map_none = {tid: tid for tid in birth_std}
            out_path = out_dir / f"{stem}_{tracker_name}_none.mp4"
            render_video(frames, fd_std, id_map_none, tracker_name, "none",
                         out_path, src_fps, skip, len(birth_std),
                         raw_std["rescue_count"], det_times_std, reid_model=reid_model)
            m = compute_metrics(tracker_name, "none", fd_std, id_map_none,
                                 birth_std, death_std, raw_std, gt_persons, pass1_s)
            results.append(m)

        if "de" in postproc_list:
            id_map_de = build_de_idmap(birth_std, death_std, embs_std, gt_persons)
            out_path = out_dir / f"{stem}_{tracker_name}_de.mp4"
            eff_frags = len({id_map_de.get(t, t) for t in birth_std})
            render_video(frames, fd_std, id_map_de, tracker_name, "de",
                         out_path, src_fps, skip, eff_frags,
                         raw_std["rescue_count"], det_times_std, reid_model=reid_model)
            m = compute_metrics(tracker_name, "de", fd_std, id_map_de,
                                 birth_std, death_std, raw_std, gt_persons, pass1_s)
            results.append(m)

    # ── Pass 2: liberal (ABC) → abc + ABCDE ──────────────────────────────────
    need_lib = any(p in postproc_list for p in ("abc", "abcde"))
    if need_lib:
        t_start = time.perf_counter()
        fd_abc, embs_abc, birth_abc, death_abc, raw_abc = tracking_pass(
            frames, tracker_name, detector, reid_ext, liberal=True)
        pass2_s = time.perf_counter() - t_start

        det_times_abc = [0.0] * len(frames)

        if "abc" in postproc_list:
            id_map_abc = {tid: tid for tid in birth_abc}
            out_path = out_dir / f"{stem}_{tracker_name}_abc.mp4"
            render_video(frames, fd_abc, id_map_abc, tracker_name, "abc",
                         out_path, src_fps, skip, len(birth_abc),
                         raw_abc["rescue_count"], det_times_abc, reid_model=reid_model)
            m = compute_metrics(tracker_name, "abc", fd_abc, id_map_abc,
                                 birth_abc, death_abc, raw_abc, gt_persons, pass2_s)
            results.append(m)

        if "abcde" in postproc_list:
            id_map_abcde = build_de_idmap(birth_abc, death_abc, embs_abc, gt_persons)
            out_path = out_dir / f"{stem}_{tracker_name}_abcde.mp4"
            eff_frags = len({id_map_abcde.get(t, t) for t in birth_abc})
            render_video(frames, fd_abc, id_map_abcde, tracker_name, "abcde",
                         out_path, src_fps, skip, eff_frags,
                         raw_abc["rescue_count"], det_times_abc, reid_model=reid_model)
            m = compute_metrics(tracker_name, "abcde", fd_abc, id_map_abcde,
                                 birth_abc, death_abc, raw_abc, gt_persons, pass2_s)
            results.append(m)

    return results


# ── summary table ──────────────────────────────────────────────────────────────

def print_summary(all_results: list[dict], gt_persons: int) -> None:
    print()
    sep = "─" * 105
    print(sep)
    if gt_persons > 0:
        sorted_r = sorted(all_results, key=lambda x: -x.get("gt_score", 0))
        print(f"  {'Rank':<5} {'Tracker':<22} {'Post':<7} │ {'GTscore':>7} │ "
              f"{'IDprec':>6} │ {'Cover':>5} │ {'ODrate':>6} │ {'Frags':>5} │ "
              f"{'Rescue':>6} │ {'Tot s':>6}")
        print(sep)
        for rank, m in enumerate(sorted_r, 1):
            print(f"  #{rank:<4} {m['tracker']:<22} {m['postproc']:<7} │ "
                  f"{m.get('gt_score', 0):>7.3f} │ "
                  f"{m.get('id_precision', 0):>6.3f} │ "
                  f"{m.get('coverage', 0):>5.3f} │ "
                  f"{m.get('od_rate', 0):>5.2f}x │ "
                  f"{m['frags']:>5} │ "
                  f"{m['rescue_count']:>6} │ "
                  f"{m['total_s']:>6.1f}")
    else:
        sorted_r = sorted(all_results, key=lambda x: x["frags"])
        print(f"  {'Rank':<5} {'Tracker':<22} {'Post':<7} │ {'Frags':>5} │ "
              f"{'Rescue':>6} │ {'AvgSim':>6} │ {'Active':>6} │ {'Tot s':>6}")
        print(sep)
        for rank, m in enumerate(sorted_r, 1):
            print(f"  #{rank:<4} {m['tracker']:<22} {m['postproc']:<7} │ "
                  f"{m['frags']:>5} │ "
                  f"{m['rescue_count']:>6} │ "
                  f"{m['avg_rescue_sim']:>6.3f} │ "
                  f"{m['avg_active']:>6.2f} │ "
                  f"{m['total_s']:>6.1f}")
    print(sep)


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video",       default="assets/test2.mp4")
    parser.add_argument("--out_dir",     default="outputs/all_combos")
    parser.add_argument("--trackers",    nargs="+", default=ALL_TRACKERS)
    parser.add_argument("--postproc",    nargs="+", default=ALL_POSTPROC,
                        choices=ALL_POSTPROC)
    parser.add_argument("--gt_persons",  type=int,  default=7)
    parser.add_argument("--frames",      type=int,  default=0,
                        help="Max frames (0=all)")
    parser.add_argument("--skip",        type=int,  default=2)
    parser.add_argument("--yolox_model", default="yolox_s")
    parser.add_argument("--reid_model",  default="osnet_x0_25",
                        choices=["osnet_x0_25", "osnet_x0_5", "osnet_x0_75",
                                 "osnet_x1_0", "osnet_ibn_x1_0", "osnet_ain_x1_0"],
                        help="Re-ID backbone (default: osnet_x0_25, stronger: osnet_x1_0)")
    parser.add_argument("--device",      default="cpu")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── load frames ───────────────────────────────────────────────────────────
    print(f"\nLoading video: {args.video}")
    cap = cv2.VideoCapture(args.video)
    assert cap.isOpened()
    src_fps   = cap.get(cv2.CAP_PROP_FPS)
    total_src = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    orig_w    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"  {orig_w}×{orig_h}  {src_fps:.1f}fps  ({total_src} frames total)")

    frames: list[np.ndarray] = []
    fid = 0;  max_f = args.frames if args.frames > 0 else 10 ** 9
    while len(frames) < max_f:
        ret, frame = cap.read()
        if not ret:
            break
        fid += 1
        if fid % args.skip != 0:
            continue
        frames.append(frame)
    cap.release()
    stem = Path(args.video).stem
    print(f"  → {len(frames)} frames loaded (skip={args.skip})")

    # ── load models ───────────────────────────────────────────────────────────
    print(f"\nLoading YOLOX ({args.yolox_model}) …")
    detector = YOLOXDetector(model_name=args.yolox_model, device=args.device,
                             high_score_thresh=0.45, low_score_thresh=0.10,
                             nms_thresh=0.45)
    detector.load_model()
    print("  → detector ready")

    print(f"Loading Re-ID ({args.reid_model}) …")
    reid_ext = FastReIDExtractor(model_path=args.reid_model, device=args.device,
                                 input_size=(256, 128), use_onnx=False)
    reid_ext.load_model()
    print(f"  → reid ready (feature_dim={reid_ext.feature_dim})")

    # ── run all combos ────────────────────────────────────────────────────────
    n_trackers = len(args.trackers)
    n_pp       = len(args.postproc)
    n_total    = n_trackers * n_pp
    print(f"\nRunning {n_trackers} trackers × {n_pp} post-proc = {n_total} combos "
          f"× {len(frames)} frames …\n")

    all_results: list[dict] = []
    combo_idx = 0

    for t_idx, trk in enumerate(args.trackers, 1):
        print(f"  [{t_idx}/{n_trackers}] {trk}", flush=True)
        try:
            t0 = time.perf_counter()
            results = process_tracker(
                frames, trk, detector, reid_ext,
                out_dir, stem, src_fps, args.skip,
                args.gt_persons, args.postproc,
                reid_model=args.reid_model,
            )
            elapsed = time.perf_counter() - t0
            for m in results:
                combo_idx += 1
                score_str = (f"  GTscore={m['gt_score']:.3f}" if args.gt_persons > 0 else "")
                print(f"    [{combo_idx:>3}/{n_total}] {trk}+{m['postproc']:<6}  "
                      f"frags={m['frags']:>3}  rescue={m['rescue_count']:>3}{score_str}")
            all_results.extend(results)
            print(f"    → done in {elapsed:.1f}s")
        except Exception as e:
            print(f"    ✗ ERROR: {e}")
            import traceback; traceback.print_exc()

    # ── summary & save ────────────────────────────────────────────────────────
    print_summary(all_results, args.gt_persons)

    json_path = out_dir / "eval_all_combos.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nVideos  → {out_dir}/  ({len(all_results)} files)")
    print(f"Metrics → {json_path}")


if __name__ == "__main__":
    main()
