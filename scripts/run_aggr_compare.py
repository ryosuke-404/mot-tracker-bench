"""
「どれだけ積極的に同一IDとして扱うか」を4段階で比較する。

プロファイル:
  default   : 現行設定（保守的）
  iou_aggr  : IoU一致率を下げて「少しでも重なればOK」
  reid_aggr : Re-ID類似度閾値を下げて「似ていれば同一人物」
  full_aggr : IoU + Re-ID 両方を積極化 + 長期バッファ

評価指標（追加）:
  GTscore     : 総合スコア（既存）
  frags       : 総ID生成数（少ないほど良い）
  avg_id_life : 1IDが生きていた平均フレーム数（長いほどID安定）
  id_survival : long-livedなID数 / GT（高いほど良い）
  rescue_rate : rescueが何件/フレームで発動したか

Usage:
    .venv/bin/python scripts/run_aggr_compare.py \\
        --video   assets/test10.mp4 \\
        --out_dir outputs/test10_aggr \\
        --gt_persons 10 \\
        --skip 2
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from detection.yolox_detector import YOLOXDetector
from tracking.bytetrack import Detection as TrackDetection
from tracking.hybridsort import HybridSORTTracker
from tracking.track import STrack, TrackState
from reid.fastreid_extractor import FastReIDExtractor
from reid.gallery import ReIDGallery
from tracking.postprocess import compute_mean_embeddings, merge_by_appearance, cluster_reassign

# ── 4プロファイル定義 ─────────────────────────────────────────────────────────
#
#  match_thresh    : コスト行列の上限閾値。低いほど「多少ズレても同一人物」
#  track_buffer    : ロスト後に何フレームIDを保持するか。長いほど復帰しやすい
#  min_hits        : 何フレーム連続検出でconfirmedにするか。低いほど早く確定
#  reid_weight     : HybridSORTのコスト: Re-IDの重み。高いほど外観優先
#  gallery_thresh  : Re-IDギャラリーのrescue閾値。低いほど積極的にrescue
#  de_sim_thresh   : DE後処理のマージ閾値。低いほど積極マージ
#  gallery_max_emb : 1IDあたりの最大保存embedding数。多いほど安定
#
PROFILES: dict[str, dict] = {
    "default": dict(
        match_thresh   = 0.90,
        track_buffer   = 180,
        min_hits       = 3,
        reid_weight    = 0.30,
        gallery_thresh = 0.72,
        de_sim_thresh  = 0.72,
        gallery_max_emb= 8,
        label          = "DEFAULT",
        desc           = "保守的・現行設定",
        color          = (150, 150, 150),   # グレー
    ),
    "iou_aggr": dict(
        match_thresh   = 0.70,   # ← 0.90→0.70: 少しでも重なればOK
        track_buffer   = 300,   # ← 180→300: 長く待ってから消す
        min_hits       = 1,     # ← 3→1: 1フレームで即確定
        reid_weight    = 0.30,
        gallery_thresh = 0.72,
        de_sim_thresh  = 0.72,
        gallery_max_emb= 8,
        label          = "IoU-AGGR",
        desc           = "IoU積極化（低閾値+長バッファ）",
        color          = (0, 200, 255),    # 水色
    ),
    "reid_aggr": dict(
        match_thresh   = 0.90,
        track_buffer   = 180,
        min_hits       = 3,
        reid_weight    = 0.50,   # ← 0.30→0.50: Re-IDを重視
        gallery_thresh = 0.55,   # ← 0.72→0.55: 似ていれば同一人物
        de_sim_thresh  = 0.55,   # ← 0.72→0.55: DEも積極マージ
        gallery_max_emb= 16,    # ← 8→16: より多くの外観を記憶
        label          = "ReID-AGGR",
        desc           = "Re-ID積極化（低類似度閾値）",
        color          = (0, 140, 255),    # オレンジ
    ),
    "full_aggr": dict(
        match_thresh   = 0.65,   # ← IoU最積極
        track_buffer   = 360,   # ← 6秒分バッファ
        min_hits       = 1,
        reid_weight    = 0.50,
        gallery_thresh = 0.50,   # ← Re-ID最積極
        de_sim_thresh  = 0.50,
        gallery_max_emb= 20,
        label          = "FULL-AGGR",
        desc           = "IoU+Re-ID 両方最積極化",
        color          = (0, 80, 255),     # 赤系
    ),
}

PROFILE_ORDER = ["default", "iou_aggr", "reid_aggr", "full_aggr"]

_STATE_COLORS = {
    TrackState.Tentative: (200, 200,   0),
    TrackState.Confirmed: (  0, 230,   0),
    TrackState.Lost:      (  0, 120, 255),
}


class FT(NamedTuple):
    track_id: int
    tlbr:     np.ndarray
    state:    TrackState


# ── 評価指標計算 ──────────────────────────────────────────────────────────────
def compute_metrics(
    all_tids: set,
    track_birth: dict,
    track_death: dict,
    active_cnts: list,
    rescue_count: int,
    n_frames: int,
    gt: int,
) -> dict:
    long_lived  = sum(1 for tid in all_tids
                      if (track_death.get(tid,0) - track_birth.get(tid,0)) >= n_frames * 0.5)
    avg_active  = float(np.mean(active_cnts)) if active_cnts else 0.0

    # 既存スコア
    id_prec  = min(gt / max(len(all_tids), 1), 1.0)
    coverage = min(long_lived / gt, 1.0)
    od_rate  = avg_active / gt
    under_p  = min(1.0 / max(od_rate, 0.01), 1.0)
    comps    = [id_prec, coverage, under_p]
    denom    = sum(1.0 / max(c, 1e-6) for c in comps)
    gt_score = len(comps) / denom if denom > 0 else 0.0

    # 追加指標
    lifetimes    = [track_death.get(t,0) - track_birth.get(t,0) for t in all_tids]
    avg_id_life  = float(np.mean(lifetimes)) if lifetimes else 0.0
    id_survival  = long_lived / gt if gt > 0 else 0.0
    rescue_rate  = rescue_count / max(n_frames, 1)

    return dict(
        gt_score    = gt_score,
        id_precision= id_prec,
        coverage    = coverage,
        od_rate     = od_rate,
        frags       = len(all_tids),
        long_lived  = long_lived,
        avg_id_life = avg_id_life,
        id_survival = id_survival,
        rescue_count= rescue_count,
        rescue_rate = rescue_rate,
        avg_active  = avg_active,
    )


# ── 描画 ──────────────────────────────────────────────────────────────────────
def draw_frame(
    frame:   np.ndarray,
    ft_list: list,
    id_map:  dict,
    profile: str,
    fid:     int,
    fps:     float,
    metrics: dict | None,
    gt:      int,
    rescue_flash: int,    # 残りフラッシュフレーム数
) -> np.ndarray:
    h, w  = frame.shape[:2]
    out   = frame.copy()
    sf    = h / 720
    pcfg  = PROFILES[profile]
    color = pcfg["color"]

    # ── バウンディングボックス ─────────────────────────────────────────────────
    for ft in ft_list:
        disp_id = id_map.get(ft.track_id, ft.track_id)
        x1, y1, x2, y2 = ft.tlbr.astype(int)
        sc = _STATE_COLORS.get(ft.state, (180, 180, 180))
        cv2.rectangle(out, (x1, y1), (x2, y2), sc, 3)
        bh = max(y2 - y1, 1)
        fs = max(0.5, min(2.0, bh / 160))
        th = max(1, int(fs * 1.5))
        lbl = f" ID {disp_id} "
        font = cv2.FONT_HERSHEY_DUPLEX
        (tw, tth), bl = cv2.getTextSize(lbl, font, fs, th)
        by2 = max(tth + bl + 8, y1);  by1_b = max(0, y1 - tth - bl - 8)
        cv2.rectangle(out, (x1, by1_b), (x1 + tw, by2), sc, cv2.FILLED)
        cv2.putText(out, lbl, (x1, by2 - bl - 3), font, fs, (0,0,0), th, cv2.LINE_AA)

    # ── rescue フラッシュ ─────────────────────────────────────────────────────
    if rescue_flash > 0:
        alpha = min(0.25, rescue_flash / 15 * 0.25)
        ovl = out.copy()
        cv2.rectangle(ovl, (0, 0), (w, h), (0, 255, 120), cv2.FILLED)
        cv2.addWeighted(ovl, alpha, out, 1 - alpha, 0, out)

    # ── 上部バナー（プロファイル名） ───────────────────────────────────────────
    tf   = cv2.FONT_HERSHEY_DUPLEX
    t_fs = 1.3 * sf;  t_th = max(2, int(sf * 2.5))
    tag  = f"  {pcfg['label']}  "
    sub  = f"  {pcfg['desc']}  "
    (tw1, th1), bl1 = cv2.getTextSize(tag, tf, t_fs, t_th)
    (tw2, th2), bl2 = cv2.getTextSize(sub, tf, t_fs * 0.65, t_th - 1)
    bw   = max(tw1, tw2) + int(40 * sf)
    bh   = th1 + th2 + bl1 + bl2 + int(24 * sf)
    bx1  = (w - bw) // 2;  by1  = int(8 * sf)
    ovl2 = out.copy()
    cv2.rectangle(ovl2, (bx1, by1), (bx1 + bw, by1 + bh), color, cv2.FILLED)
    cv2.addWeighted(ovl2, 0.88, out, 0.12, 0, out)
    pad = int(10 * sf)
    cv2.putText(out, tag, ((w - tw1) // 2, by1 + pad + th1),
                tf, t_fs, (0,0,0), t_th, cv2.LINE_AA)
    cv2.putText(out, sub, ((w - tw2) // 2, by1 + pad + th1 + bl1 + th2 + int(4*sf)),
                tf, t_fs * 0.65, (0,0,0), max(1, t_th-1), cv2.LINE_AA)

    # ── パラメータ表示（左上） ────────────────────────────────────────────────
    sf_font = cv2.FONT_HERSHEY_DUPLEX
    sf_fs   = 0.50 * sf;  sf_th = max(1, int(sf))
    lh      = int(24 * sf);  pd = int(10 * sf)
    params  = [
        f"match_thresh  : {pcfg['match_thresh']}",
        f"track_buffer  : {pcfg['track_buffer']}",
        f"min_hits      : {pcfg['min_hits']}",
        f"reid_weight   : {pcfg['reid_weight']}",
        f"gallery_thresh: {pcfg['gallery_thresh']}",
        f"de_sim_thresh : {pcfg['de_sim_thresh']}",
        f"──────────────────",
        f"Frame : {fid}   FPS: {fps:.1f}",
        f"Active: {sum(1 for ft in ft_list if ft.state == TrackState.Confirmed)}",
    ]
    (pw, _), _ = cv2.getTextSize("gallery_thresh: 0.55", sf_font, sf_fs, sf_th)
    pw2 = pw + pd * 2;  ph = lh * len(params) + pd * 2
    ovl3 = out.copy()
    cv2.rectangle(ovl3, (0, 0), (pw2, ph), (10, 10, 10), cv2.FILLED)
    cv2.addWeighted(ovl3, 0.65, out, 0.35, 0, out)
    for i, text in enumerate(params):
        col = (0, 255, 200) if "──" in text else (200, 200, 200)
        cv2.putText(out, text, (pd, pd + lh * (i+1) - int(3*sf)),
                    sf_font, sf_fs, col, sf_th, cv2.LINE_AA)

    # ── 評価スコアパネル（右下・大） ───────────────────────────────────────────
    if metrics:
        score   = metrics["gt_score"]
        frags   = metrics["frags"]
        alife   = metrics["avg_id_life"]
        surv    = metrics["id_survival"]
        rcount  = metrics["rescue_count"]
        cov     = metrics["coverage"]

        # スコア色
        def scol(v, good, mid):
            if v >= good: return (0, 230, 0)
            if v >= mid:  return (0, 220, 220)
            return (80, 80, 255)

        lines = [
            ("GTscore",   f"{score:.3f}",  scol(score, 0.7, 0.4),  1.15),
            ("Coverage",  f"{cov:.3f}",    scol(cov,   0.8, 0.5),  0.80),
            ("IDsurv",    f"{surv:.3f}",   scol(surv,  0.8, 0.5),  0.80),
            ("frags",     f"{frags}  (GT={gt})", scol(gt/max(frags,1), 0.7, 0.4), 0.80),
            ("avg life",  f"{alife:.0f}f", scol(alife/max(len(frames_ref[0]),1) if frames_ref else 0, 0.3, 0.1), 0.72),
            ("rescue",    f"{rcount}回",   (180, 180, 180),          0.72),
        ]

        ef   = cv2.FONT_HERSHEY_DUPLEX
        base_fs = 0.85 * sf
        base_th = max(1, int(sf * 1.5))
        lhe  = int(36 * sf);  pde = int(14 * sf)

        max_lw = 0
        for lbl, val, col, fmul in lines:
            (lw, _), _ = cv2.getTextSize(f"{lbl} : {val}", ef, base_fs * fmul, base_th)
            max_lw = max(max_lw, lw)
        pw4  = max_lw + pde * 3
        ph4  = lhe * len(lines) + pde * 2
        px1  = w - pw4 - int(8 * sf)
        py1  = h - ph4 - int(8 * sf)

        ovl4 = out.copy()
        cv2.rectangle(ovl4, (px1 - pde, py1 - pde), (w - int(4*sf), h - int(4*sf)),
                      (10, 10, 10), cv2.FILLED)
        cv2.addWeighted(ovl4, 0.78, out, 0.22, 0, out)
        # プロファイル色のサイドライン
        cv2.rectangle(out, (px1 - pde, py1 - pde), (px1 - pde + int(5*sf), h - int(4*sf)),
                      color, cv2.FILLED)

        for i, (lbl, val, col, fmul) in enumerate(lines):
            fs_i = base_fs * fmul
            th_i = max(1, int(fs_i * 1.8))
            cv2.putText(out, f"{lbl} : {val}",
                        (px1, py1 + pde + lhe * (i+1)),
                        ef, fs_i, col, th_i, cv2.LINE_AA)
    return out


# グローバル参照用（avg_id_life計算）
frames_ref: list = []


# ── 1プロファイル実行 ─────────────────────────────────────────────────────────
def run_profile(
    frames:   list[np.ndarray],
    profile:  str,
    detector: YOLOXDetector,
    reid_ext: FastReIDExtractor,
    out_path: Path,
    src_fps:  float,
    skip:     int,
    gt:       int,
) -> dict:
    pcfg = PROFILES[profile]

    STrack.reset_id_counter()
    tracker = HybridSORTTracker(
        track_thresh     = 0.50,
        track_buffer     = pcfg["track_buffer"],
        match_thresh     = pcfg["match_thresh"],
        min_hits         = pcfg["min_hits"],
        iou_thresh_stage2= 0.40,
        iou_weight       = 0.40,
        height_weight    = 0.20,
        shape_weight     = 0.10,
        reid_weight      = pcfg["reid_weight"],
    )
    gallery = ReIDGallery(
        max_gallery_size    = 300,
        similarity_thresh   = pcfg["gallery_thresh"],
        max_embeddings_per_id = pcfg["gallery_max_emb"],
    )

    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    writer = cv2.VideoWriter(str(out_path), fourcc, src_fps / skip, (w, h))
    if not writer.isOpened():
        writer = cv2.VideoWriter(str(out_path),
                                 cv2.VideoWriter_fourcc(*"mp4v"), src_fps / skip, (w, h))

    all_tids:    set[int]       = set()
    track_birth: dict[int,int]  = {}
    track_death: dict[int,int]  = {}
    track_embs:  dict           = {}
    active_cnts: list[int]      = []
    det_times:   list[float]    = []
    rescue_count = 0
    fps_t0 = time.perf_counter();  fps_cnt = 0;  fps_disp = 0.0
    rescue_flash = 0
    metrics = None

    for fid, frame in enumerate(frames, 1):
        t0 = time.perf_counter()
        high_raw, low_raw = detector.detect(frame)
        det_times.append((time.perf_counter() - t0) * 1000)

        high_embs = reid_ext.extract_batch(frame, [d.bbox for d in high_raw]) \
            if high_raw else np.zeros((0, reid_ext.feature_dim), np.float32)
        high_dets = [TrackDetection(bbox=d.bbox, score=d.score, embedding=high_embs[i])
                     for i, d in enumerate(high_raw)]
        low_dets  = [TrackDetection(bbox=d.bbox, score=d.score) for d in low_raw]

        active     = tracker.update(high_dets, low_dets)
        active_ids = {t.track_id for t in active}
        all_tids.update(active_ids)

        confirmed = [t for t in active if t.state == TrackState.Confirmed]
        for t in confirmed:
            if t.reid_embedding is not None:
                gallery.add_embedding(t.track_id, t.reid_embedding, fid)
                track_embs.setdefault(t.track_id, []).append(t.reid_embedding)
            track_birth.setdefault(t.track_id, fid)
            track_death[t.track_id] = fid

        # Re-ID rescue（gallery_thresh適用）
        for t in active:
            if t.state == TrackState.Tentative and t.hits == 1 and t.reid_embedding is not None:
                mid, sim = gallery.query(t.reid_embedding, exclude_ids=active_ids)
                if mid is not None:
                    rescue_count += 1
                    rescue_flash = 20
                    old = t.track_id
                    t.reassign_id(mid)
                    gallery.remove_track(old)

        if fid % 150 == 0:
            lost = {t.track_id for t in tracker.get_lost_tracks()}
            gallery.prune_old_tracks(active_ids | lost, 3000, fid)

        active_cnts.append(len(active))
        fps_cnt += 1
        if fps_cnt >= 30:
            fps_disp = fps_cnt / (time.perf_counter() - fps_t0)
            fps_cnt  = 0;  fps_t0 = time.perf_counter()

        if rescue_flash > 0:
            rescue_flash -= 1

        # 中間メトリクス
        if fid % 30 == 0 or fid == len(frames):
            metrics = compute_metrics(all_tids, track_birth, track_death,
                                      active_cnts, rescue_count, fid, gt)

        fts = [FT(t.track_id, t.tlbr.copy(), t.state) for t in active]
        vis = draw_frame(frame, fts, {}, profile, fid, fps_disp,
                         metrics, gt, rescue_flash)
        writer.write(vis)

    writer.release()

    # DE後処理（full_aggrとreid_aggrはDEも適用）
    id_map: dict[int,int] = {}
    if profile in ("reid_aggr", "full_aggr"):
        from tracking.postprocess import compute_mean_embeddings
        mean_embs = compute_mean_embeddings(track_embs)
        merged    = merge_by_appearance(track_birth, track_death, mean_embs,
                                        sim_thresh=pcfg["de_sim_thresh"])
        id_map    = merged
        effective = {id_map.get(t, t) for t in all_tids}
    else:
        effective = all_tids

    final = compute_metrics(effective, track_birth, track_death,
                            active_cnts, rescue_count, len(frames), gt)
    final["profile"]     = profile
    final["label"]       = pcfg["label"]
    final["desc"]        = pcfg["desc"]
    final["params"]      = {k: v for k, v in pcfg.items()
                            if k not in ("label","desc","color")}
    return final


# ── サマリー動画（4分割） ─────────────────────────────────────────────────────
def make_summary_video(
    frames:    list[np.ndarray],
    results:   dict[str, dict],
    out_path:  Path,
    src_fps:   float,
    skip:      int,
):
    """4プロファイルの動画を2×2グリッドで並べたサマリー動画を生成"""
    h, w = frames[0].shape[:2]
    gh, gw = h // 2, w // 2   # グリッド1セルのサイズ

    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    writer = cv2.VideoWriter(str(out_path), fourcc, src_fps / skip, (w, h))
    if not writer.isOpened():
        writer = cv2.VideoWriter(str(out_path),
                                 cv2.VideoWriter_fourcc(*"mp4v"), src_fps / skip, (w, h))

    import cv2 as _cv2
    profile_order = PROFILE_ORDER
    video_readers = []
    for p in profile_order:
        vpath = out_path.parent / f"{out_path.stem.replace('_summary','_'+p)}.mp4"
        if vpath.exists():
            video_readers.append(_cv2.VideoCapture(str(vpath)))
        else:
            video_readers.append(None)

    while True:
        grid = np.zeros((h, w, 3), np.uint8)
        any_frame = False
        for i, (p, cap) in enumerate(zip(profile_order, video_readers)):
            if cap is None:
                cell = np.zeros((gh, gw, 3), np.uint8)
            else:
                ret, cell = cap.read()
                if not ret:
                    cell = np.zeros((gh, gw, 3), np.uint8)
                else:
                    any_frame = True
                    cell = _cv2.resize(cell, (gw, gh))

            # GTscoreをセル右下にオーバーレイ
            m = results.get(p)
            if m:
                score = m["gt_score"]
                scol  = (0,230,0) if score>0.7 else (0,220,220) if score>0.4 else (80,80,255)
                sf2   = gh / 720
                ef    = _cv2.FONT_HERSHEY_DUPLEX
                txt   = f"GTscore:{score:.3f}"
                fs2   = 0.9 * sf2;  th2 = max(1, int(sf2 * 1.5))
                (lw2, lh2), _ = _cv2.getTextSize(txt, ef, fs2, th2)
                _cv2.rectangle(cell, (gw - lw2 - 12, gh - lh2 - 16),
                               (gw - 2, gh - 2), (10,10,10), _cv2.FILLED)
                _cv2.putText(cell, txt, (gw - lw2 - 8, gh - 8),
                             ef, fs2, scol, th2, _cv2.LINE_AA)

            row, col = divmod(i, 2)
            grid[row*gh:(row+1)*gh, col*gw:(col+1)*gw] = cell

        if not any_frame:
            break
        writer.write(grid)

    for cap in video_readers:
        if cap: cap.release()
    writer.release()


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    global frames_ref

    parser = argparse.ArgumentParser()
    parser.add_argument("--video",       default="assets/test10.mp4")
    parser.add_argument("--out_dir",     default="outputs/test10_aggr")
    parser.add_argument("--gt_persons",  type=int, default=10)
    parser.add_argument("--frames",      type=int, default=0)
    parser.add_argument("--skip",        type=int, default=2)
    parser.add_argument("--yolox_model", default="yolox_s")
    parser.add_argument("--device",      default="cpu")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.video).stem

    # ── フレーム読み込み ───────────────────────────────────────────────────────
    print(f"\n動画読み込み: {args.video}")
    cap = cv2.VideoCapture(args.video)
    assert cap.isOpened()
    src_fps   = cap.get(cv2.CAP_PROP_FPS)
    total_src = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ow = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    oh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"  {ow}×{oh}  {src_fps:.1f}fps  ({total_src}フレーム)")

    frames: list[np.ndarray] = []
    fid = 0;  max_f = args.frames if args.frames > 0 else 10**9
    while len(frames) < max_f:
        ret, frame = cap.read()
        if not ret: break
        fid += 1
        if fid % args.skip != 0: continue
        frames.append(frame)
    cap.release()
    frames_ref.append(frames)  # avg_id_life計算用
    print(f"  → {len(frames)} フレーム (skip={args.skip})")

    # ── 検出器・Re-ID 読み込み ─────────────────────────────────────────────────
    print(f"\nYOLOX ({args.yolox_model}) 読み込み中…")
    detector = YOLOXDetector(model_name=args.yolox_model, device=args.device,
                             high_score_thresh=0.45, low_score_thresh=0.10, nms_thresh=0.45)
    detector.load_model()

    print("Re-ID (osnet_x0_25) 読み込み中…")
    reid_ext = FastReIDExtractor(model_path="osnet_x0_25", device=args.device, use_onnx=False)
    reid_ext.load_model()
    print("  → 準備完了\n")

    # ── 4プロファイル実行 ─────────────────────────────────────────────────────
    sep = "─" * 90
    print(f"4プロファイル比較 (GT={args.gt_persons}人)\n")
    print(f"{'':4} {'プロファイル':<14} {'GTscore':>8} {'frags':>6} {'avgLife':>8} "
          f"{'IDsurv':>7} {'Cover':>6} {'rescue':>7} {'秒':>6}")
    print(sep)

    all_results: dict[str, dict] = {}

    for profile in PROFILE_ORDER:
        pcfg     = PROFILES[profile]
        out_path = out_dir / f"{stem}_{profile}.mp4"
        print(f"  [{profile:<10}] {pcfg['desc']:<28}", end="  ", flush=True)
        t0 = time.perf_counter()

        try:
            m = run_profile(frames, profile, detector, reid_ext,
                            out_path, src_fps, args.skip, args.gt_persons)
            elapsed = time.perf_counter() - t0
            m["total_s"] = elapsed
            all_results[profile] = m
            print(f"✓  {m['gt_score']:.3f}  {m['frags']:>5}  "
                  f"{m['avg_id_life']:>7.1f}f  {m['id_survival']:>6.3f}  "
                  f"{m['coverage']:>5.3f}  {m['rescue_count']:>6}  {elapsed:>5.0f}s")
        except Exception as e:
            import traceback
            elapsed = time.perf_counter() - t0
            print(f"✗  ERROR: {e}  ({elapsed:.0f}s)")
            traceback.print_exc()

    # ── サマリー動画（2×2グリッド） ───────────────────────────────────────────
    print(f"\nサマリー動画（2×2グリッド）生成中…")
    summary_path = out_dir / f"{stem}_summary.mp4"
    try:
        make_summary_video(frames, all_results, summary_path, src_fps, args.skip)
        print(f"  → {summary_path}")
    except Exception as e:
        print(f"  サマリー動画エラー: {e}")

    # ── 最終比較表 ────────────────────────────────────────────────────────────
    print(f"\n{sep}")
    print("  4プロファイル 詳細比較\n")
    print(f"  {'プロファイル':<14} │ {'GTscore':>8} │ {'frags':>5} │ {'avgLife':>7} │ "
          f"{'IDsurv':>6} │ {'Cover':>5} │ {'rescue':>6}")
    print(f"  {sep}")
    for profile in PROFILE_ORDER:
        m = all_results.get(profile)
        if not m: continue
        best = "★" if m["gt_score"] == max(r["gt_score"] for r in all_results.values()) else " "
        print(f"  {best}{PROFILES[profile]['label']:<13} │ {m['gt_score']:>8.3f} │ "
              f"{m['frags']:>5} │ {m['avg_id_life']:>6.1f}f │ "
              f"{m['id_survival']:>6.3f} │ {m['coverage']:>5.3f} │ {m['rescue_count']:>6}")
    print(f"  {sep}")

    # ── 指標の説明 ────────────────────────────────────────────────────────────
    print("""
  指標説明:
    GTscore  : 総合スコア（IDprec × Coverage × ODrate の調和平均）
    frags    : 総ID生成数 ← 少ないほど「同一人物をIDキープできた」
    avgLife  : 1IDが生きていた平均フレーム数 ← 長いほど安定
    IDsurv   : 全フレームの50%以上生存したID数 / GT ← 高いほど良い
    Cover    : GT人数全員を長期追跡できたか ← 1.0が理想
    rescue   : Re-IDによるID統合回数
""")

    json_path = out_dir / "eval_aggr.json"
    with open(json_path, "w") as f:
        json.dump(list(all_results.values()), f, indent=2, ensure_ascii=False)
    print(f"  動画  → {out_dir}/  (4本 + サマリー1本)")
    print(f"  JSON  → {json_path}")


if __name__ == "__main__":
    main()
