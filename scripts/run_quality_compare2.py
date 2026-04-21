"""
画質変化によるトラッキング精度比較 — GTscore + IDF1 + HOTA 併用版

疑似GT生成:
  original画質で hybridsort_tuned+ReID-AGGR を実行 → フレームごとの
  (track_id, bbox) を疑似GTとして保存。
  他の画質プロファイルをこれに対して評価する。

評価指標:
  GTscore  : 独自指標（ID精度 × Coverage × ODrate の調和平均）
  IDF1     : Identification F1（ID一致率）
  HOTA     : Higher Order Tracking Accuracy
  DetA     : 検出精度（HOTAの検出コンポーネント）
  AssA     : 関連付け精度（HOTAの追跡コンポーネント）

Usage:
    .venv/bin/python scripts/run_quality_compare2.py \\
        --video   assets/test10.mp4 \\
        --out_dir outputs/test10_quality2 \\
        --gt_persons 10 \\
        --skip 2
"""
from __future__ import annotations

import argparse
import json
import sys
import time
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
from tracking.postprocess import compute_mean_embeddings, merge_by_appearance
from reid.mot_metrics import MOTEvaluator

# ── 画質バリエーション ────────────────────────────────────────────────────────
QUALITY_PROFILES: list[dict] = [
    dict(key="original",    label="ORIGINAL",    desc="オリジナル",
         color=(200, 200, 200)),
    dict(key="blur_mild",   label="BLUR-MILD",   desc="ぼかし σ=2",
         color=(0, 200, 255)),
    dict(key="blur_heavy",  label="BLUR-HEAVY",  desc="ぼかし σ=5",
         color=(0, 100, 255)),
    dict(key="jpeg_75",     label="JPEG-75",     desc="JPEG quality=75",
         color=(0, 200, 100)),
    dict(key="jpeg_30",     label="JPEG-30",     desc="JPEG quality=30",
         color=(0, 100, 50)),
    dict(key="half_res",    label="HALF-RES",    desc="解像度1/2",
         color=(200, 100, 0)),
    dict(key="quarter_res", label="QUARTER-RES", desc="解像度1/4",
         color=(255, 50, 0)),
]

# ReID-AGGR 設定（固定）
TRACKER_CFG = dict(
    track_thresh=0.50, track_buffer=180, match_thresh=0.90, min_hits=3,
    iou_thresh_stage2=0.40, iou_weight=0.40, height_weight=0.20,
    shape_weight=0.10, reid_weight=0.50,
)
GALLERY_CFG = dict(max_gallery_size=300, similarity_thresh=0.55,
                   max_embeddings_per_id=16)
DE_SIM_THRESH = 0.55

_STATE_COLORS = {
    TrackState.Tentative: (200, 200,  0),
    TrackState.Confirmed: (  0, 230,  0),
    TrackState.Lost:      (  0, 120, 255),
}


class FT(NamedTuple):
    track_id: int
    tlbr:     np.ndarray
    state:    TrackState


# ── 画質変換 ──────────────────────────────────────────────────────────────────
def degrade(frame: np.ndarray, key: str) -> np.ndarray:
    h, w = frame.shape[:2]
    if key == "original":    return frame
    if key == "blur_mild":   return cv2.GaussianBlur(frame, (0,0), 2)
    if key == "blur_heavy":  return cv2.GaussianBlur(frame, (0,0), 5)
    if key == "jpeg_75":
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if key == "jpeg_30":
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 30])
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if key == "half_res":
        s = cv2.resize(frame, (w//2, h//2), interpolation=cv2.INTER_AREA)
        return cv2.resize(s, (w, h), interpolation=cv2.INTER_LINEAR)
    if key == "quarter_res":
        s = cv2.resize(frame, (w//4, h//4), interpolation=cv2.INTER_AREA)
        return cv2.resize(s, (w, h), interpolation=cv2.INTER_LINEAR)
    return frame


# ── GTscore 計算 ──────────────────────────────────────────────────────────────
def compute_gtscore(all_tids, track_birth, track_death, active_cnts,
                    rescue_count, n_frames, gt) -> dict:
    long_lived = sum(1 for t in all_tids
                     if (track_death.get(t,0) - track_birth.get(t,0)) >= n_frames * 0.5)
    avg_active = float(np.mean(active_cnts)) if active_cnts else 0.0
    id_prec    = min(gt / max(len(all_tids), 1), 1.0)
    coverage   = min(long_lived / gt, 1.0)
    od_rate    = avg_active / gt
    under_p    = min(1.0 / max(od_rate, 0.01), 1.0)
    comps      = [id_prec, coverage, under_p]
    denom      = sum(1.0 / max(c, 1e-6) for c in comps)
    gt_score   = len(comps) / denom if denom > 0 else 0.0
    lifetimes  = [track_death.get(t,0) - track_birth.get(t,0) for t in all_tids]
    return dict(
        gt_score    = gt_score,
        id_precision= id_prec,
        coverage    = coverage,
        frags       = len(all_tids),
        avg_id_life = float(np.mean(lifetimes)) if lifetimes else 0.0,
        id_survival = long_lived / gt if gt > 0 else 0.0,
        rescue_count= rescue_count,
        avg_active  = avg_active,
    )


# ── 描画 ──────────────────────────────────────────────────────────────────────
def draw_frame(orig: np.ndarray, deg: np.ndarray, fts: list, qcfg: dict,
               fid: int, fps: float, det_ms: float,
               metrics: dict | None, gt: int, n_frames: int) -> np.ndarray:
    out = orig.copy()
    h, w = out.shape[:2]
    sf   = h / 720
    color= qcfg["color"]

    # バウンディングボックス
    for ft in fts:
        x1,y1,x2,y2 = ft.tlbr.astype(int)
        sc = _STATE_COLORS.get(ft.state, (180,180,180))
        cv2.rectangle(out, (x1,y1), (x2,y2), sc, 3)
        bh = max(y2-y1, 1)
        fs = max(0.5, min(2.0, bh/160))
        th = max(1, int(fs*1.5))
        lbl = f" ID {ft.track_id} "
        font= cv2.FONT_HERSHEY_DUPLEX
        (tw,tth),bl = cv2.getTextSize(lbl, font, fs, th)
        by2= max(tth+bl+8, y1);  by1b= max(0, y1-tth-bl-8)
        cv2.rectangle(out, (x1,by1b), (x1+tw,by2), sc, cv2.FILLED)
        cv2.putText(out, lbl, (x1, by2-bl-3), font, fs, (0,0,0), th, cv2.LINE_AA)

    # 入力画像プレビュー（左下）
    ph = int(h*0.20);  pw2 = int(ph*w/h)
    prev = cv2.resize(deg, (pw2, ph))
    px, py = int(8*sf), h-ph-int(8*sf)
    cv2.rectangle(out, (px-2,py-2), (px+pw2+2,py+ph+2), color, 2)
    out[py:py+ph, px:px+pw2] = prev
    cv2.putText(out, "INPUT", (px+4, py+int(14*sf)),
                cv2.FONT_HERSHEY_DUPLEX, 0.42*sf, color, max(1,int(sf)), cv2.LINE_AA)

    # 上部バナー
    tf  = cv2.FONT_HERSHEY_DUPLEX
    tfs = 1.2*sf;  tth2 = max(2, int(sf*2.5))
    tag = f"  {qcfg['label']}  ";  sub = f"  {qcfg['desc']}  "
    (tw1,th1),_ = cv2.getTextSize(tag, tf, tfs, tth2)
    (tw2,th2),_ = cv2.getTextSize(sub, tf, tfs*0.7, tth2-1)
    bw = max(tw1,tw2)+int(40*sf);  bh2= th1+th2+int(28*sf)
    bx1= (w-bw)//2;  by1 = int(8*sf)
    ovl= out.copy()
    cv2.rectangle(ovl, (bx1,by1), (bx1+bw,by1+bh2), color, cv2.FILLED)
    cv2.addWeighted(ovl, 0.88, out, 0.12, 0, out)
    pad= int(10*sf)
    cv2.putText(out, tag, ((w-tw1)//2, by1+pad+th1), tf, tfs, (0,0,0), tth2, cv2.LINE_AA)
    cv2.putText(out, sub, ((w-tw2)//2, by1+pad+th1+int(4*sf)+th2),
                tf, tfs*0.7, (0,0,0), max(1,tth2-1), cv2.LINE_AA)

    # 右下スコアパネル（GTscore + IDF1 + HOTA）
    if metrics:
        def scol(v, g, m):
            return (0,230,0) if v>=g else (0,220,220) if v>=m else (80,80,255)

        lines = [
            # label,  value,   color,           font_mul
            ("GTscore", f"{metrics.get('gt_score',0):.3f}",
             scol(metrics.get('gt_score',0), 0.7, 0.4), 1.10),
            ("IDF1",    f"{metrics.get('idf1',0):.3f}",
             scol(metrics.get('idf1',0), 0.7, 0.4), 1.10),
            ("HOTA",    f"{metrics.get('hota',0):.3f}",
             scol(metrics.get('hota',0), 0.6, 0.3), 1.10),
            ("DetA",    f"{metrics.get('deta',0):.3f}", (180,180,180), 0.75),
            ("AssA",    f"{metrics.get('assa',0):.3f}", (180,180,180), 0.75),
            ("frags",   f"{metrics.get('frags',0)} / GT={gt}",
             scol(gt/max(metrics.get('frags',1),1), 0.7, 0.4), 0.75),
            ("Coverage",f"{metrics.get('coverage',0):.3f}", (200,200,200), 0.72),
        ]

        ef   = cv2.FONT_HERSHEY_DUPLEX
        bfs  = 0.82*sf;  bth = max(1, int(sf*1.5))
        lhe  = int(34*sf);  pde = int(12*sf)
        max_lw = 0
        for lbl,val,_,fmul in lines:
            (lw,_),_ = cv2.getTextSize(f"{lbl} : {val}", ef, bfs*fmul, bth)
            max_lw = max(max_lw, lw)
        pw3 = max_lw+pde*3;  ph3 = lhe*len(lines)+pde*2
        ex1 = w-pw3-int(8*sf);  ey1 = h-ph3-int(8*sf)
        ovl2= out.copy()
        cv2.rectangle(ovl2, (ex1-pde,ey1-pde), (w-int(4*sf),h-int(4*sf)),
                      (10,10,10), cv2.FILLED)
        cv2.addWeighted(ovl2, 0.78, out, 0.22, 0, out)
        cv2.rectangle(out, (ex1-pde,ey1-pde), (ex1-pde+int(5*sf),h-int(4*sf)),
                      color, cv2.FILLED)
        for i,(lbl,val,col,fmul) in enumerate(lines):
            fs_i = bfs*fmul;  th_i = max(1, int(fs_i*1.8))
            cv2.putText(out, f"{lbl} : {val}", (ex1, ey1+pde+lhe*(i+1)),
                        ef, fs_i, col, th_i, cv2.LINE_AA)

    # フレーム情報（右上）
    info = f"Frame {fid}/{n_frames}  {fps:.1f}fps  {det_ms:.0f}ms"
    (iw,_),_ = cv2.getTextSize(info, cv2.FONT_HERSHEY_DUPLEX, 0.5*sf, max(1,int(sf)))
    cv2.putText(out, info, (w-iw-int(10*sf), int(20*sf)),
                cv2.FONT_HERSHEY_DUPLEX, 0.5*sf, (200,200,200), max(1,int(sf)), cv2.LINE_AA)
    return out


# ── トラッキング実行（フレームデータも返す） ──────────────────────────────────
def run_tracking(
    frames:   list[np.ndarray],
    quality_key: str,
    detector: YOLOXDetector,
    reid_ext: FastReIDExtractor,
    gt:       int,
) -> tuple[list[list[tuple[int, np.ndarray]]], dict]:
    """
    Returns:
        frame_tracks: フレームごとの [(track_id, bbox_xyxy), ...]
        raw_metrics:  GTscore等（IDF1/HOTA除く）
    """
    STrack.reset_id_counter()
    tracker = HybridSORTTracker(**TRACKER_CFG)
    gallery = ReIDGallery(**GALLERY_CFG)

    all_tids:    set[int]      = set()
    track_birth: dict[int,int] = {}
    track_death: dict[int,int] = {}
    track_embs:  dict          = {}
    active_cnts: list[int]     = []
    det_times:   list[float]   = []
    rescue_count = 0
    frame_tracks: list[list[tuple[int, np.ndarray]]] = []

    for fid, orig_frame in enumerate(frames, 1):
        deg_frame = degrade(orig_frame, quality_key)
        t0 = time.perf_counter()
        high_raw, low_raw = detector.detect(deg_frame)
        det_times.append((time.perf_counter() - t0) * 1000)

        high_embs = reid_ext.extract_batch(deg_frame, [d.bbox for d in high_raw]) \
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

        for t in active:
            if t.state == TrackState.Tentative and t.hits == 1 \
                    and t.reid_embedding is not None:
                mid, sim = gallery.query(t.reid_embedding, exclude_ids=active_ids)
                if mid is not None:
                    rescue_count += 1
                    old = t.track_id
                    t.reassign_id(mid)
                    gallery.remove_track(old)

        if fid % 150 == 0:
            lost = {t.track_id for t in tracker.get_lost_tracks()}
            gallery.prune_old_tracks(active_ids | lost, 3000, fid)

        active_cnts.append(len(active))

        # フレームデータ保存（confirmed のみ）
        fdata = [(t.track_id, t.tlbr.copy()) for t in active
                 if t.state == TrackState.Confirmed]
        frame_tracks.append(fdata)

    # DE後処理
    mean_embs = compute_mean_embeddings(track_embs)
    merged    = merge_by_appearance(track_birth, track_death, mean_embs,
                                    sim_thresh=DE_SIM_THRESH)
    # DE後のIDを反映
    frame_tracks_de = [
        [(merged.get(tid, tid), bbox) for tid, bbox in fdata]
        for fdata in frame_tracks
    ]
    effective = {merged.get(t,t) for t in all_tids}

    raw = compute_gtscore(effective, track_birth, track_death,
                          active_cnts, rescue_count, len(frames), gt)
    raw["avg_det_ms"] = float(np.mean(det_times)) if det_times else 0.0

    return frame_tracks_de, raw, det_times


# ── 動画書き出し ──────────────────────────────────────────────────────────────
def write_video(
    frames:       list[np.ndarray],
    frame_tracks: list[list],
    qcfg:         dict,
    out_path:     Path,
    src_fps:      float,
    skip:         int,
    metrics:      dict,
    gt:           int,
    det_times:    list[float],
) -> None:
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    writer = cv2.VideoWriter(str(out_path), fourcc, src_fps/skip, (w,h))
    if not writer.isOpened():
        writer = cv2.VideoWriter(str(out_path),
                                 cv2.VideoWriter_fourcc(*"mp4v"), src_fps/skip, (w,h))

    for fid, (orig, fdata) in enumerate(zip(frames, frame_tracks), 1):
        deg  = degrade(orig, qcfg["key"])
        fts  = [FT(tid, bbox, TrackState.Confirmed) for tid, bbox in fdata]
        vis  = draw_frame(orig, deg, fts, qcfg, fid, 0.0,
                          det_times[fid-1] if fid-1 < len(det_times) else 0.0,
                          metrics, gt, len(frames))
        writer.write(vis)
    writer.release()


# ── サマリー動画（グリッド） ──────────────────────────────────────────────────
def make_summary(out_dir: Path, stem: str, src_fps: float, skip: float,
                 results: list[dict]) -> Path:
    n    = len(QUALITY_PROFILES)
    cols = 3;  rows = (n+cols-1)//cols
    first_path = out_dir / f"{stem}_{QUALITY_PROFILES[0]['key']}.mp4"
    cap0 = cv2.VideoCapture(str(first_path))
    fw   = int(cap0.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh   = int(cap0.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap0.release()
    gw = fw//cols;  gh = fh//rows
    out_w = gw*cols;  out_h = gh*rows

    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    out_path = out_dir / f"{stem}_summary.mp4"
    writer   = cv2.VideoWriter(str(out_path), fourcc, src_fps/skip, (out_w, out_h))
    if not writer.isOpened():
        writer = cv2.VideoWriter(str(out_path),
                                 cv2.VideoWriter_fourcc(*"mp4v"), src_fps/skip, (out_w, out_h))

    caps = [cv2.VideoCapture(str(out_dir/f"{stem}_{q['key']}.mp4"))
            for q in QUALITY_PROFILES]
    score_map = {r["key"]: r for r in results}

    while True:
        grid = np.zeros((out_h, out_w, 3), np.uint8)
        any_frame = False
        for i, (qcfg, cap) in enumerate(zip(QUALITY_PROFILES, caps)):
            row, col = divmod(i, cols)
            ret, cell = cap.read()
            if ret:
                any_frame = True
                cell = cv2.resize(cell, (gw, gh))
            else:
                cell = np.zeros((gh, gw, 3), np.uint8)

            # スコアオーバーレイ（GTscore / IDF1 / HOTA）
            m = score_map.get(qcfg["key"], {})
            sf2 = gh/720;  ef = cv2.FONT_HERSHEY_DUPLEX
            for j, (key2, label2, good) in enumerate([
                ("gt_score","GTscore",0.7),
                ("idf1",    "IDF1",   0.7),
                ("hota",    "HOTA",   0.6),
            ]):
                v = m.get(key2, 0)
                sc2 = (0,230,0) if v>=good else (0,220,220) if v>=good*0.6 else (80,80,255)
                txt = f"{label2}:{v:.3f}"
                fs2 = 0.65*sf2;  th2 = max(1,int(sf2*1.2))
                (lw2,lh2),_ = cv2.getTextSize(txt, ef, fs2, th2)
                yy = lh2+6 + j*(lh2+4)
                cv2.rectangle(cell, (2,yy-lh2-2), (lw2+8,yy+4), (10,10,10), cv2.FILLED)
                cv2.putText(cell, txt, (5,yy), ef, fs2, sc2, th2, cv2.LINE_AA)

            grid[row*gh:(row+1)*gh, col*gw:(col+1)*gw] = cell
        if not any_frame:
            break
        writer.write(grid)

    for cap in caps: cap.release()
    writer.release()
    return out_path


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video",       default="assets/test10.mp4")
    parser.add_argument("--out_dir",     default="outputs/test10_quality2")
    parser.add_argument("--gt_persons",  type=int, default=10)
    parser.add_argument("--frames",      type=int, default=0)
    parser.add_argument("--skip",        type=int, default=2)
    parser.add_argument("--yolox_model", default="yolox_s")
    parser.add_argument("--device",      default="cpu")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.video).stem

    # フレーム読み込み
    print(f"\n動画読み込み: {args.video}")
    cap = cv2.VideoCapture(args.video)
    assert cap.isOpened()
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    total   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ow = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    oh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"  {ow}×{oh}  {src_fps:.1f}fps  ({total}フレーム)")

    frames: list[np.ndarray] = []
    fid = 0;  max_f = args.frames if args.frames > 0 else 10**9
    while len(frames) < max_f:
        ret, frame = cap.read()
        if not ret: break
        fid += 1
        if fid % args.skip != 0: continue
        frames.append(frame)
    cap.release()
    print(f"  → {len(frames)} フレーム (skip={args.skip})")

    # 検出器・Re-ID
    print(f"\nYOLOX ({args.yolox_model}) 読み込み中…")
    detector = YOLOXDetector(model_name=args.yolox_model, device=args.device,
                             high_score_thresh=0.45, low_score_thresh=0.10, nms_thresh=0.45)
    detector.load_model()
    print("Re-ID (osnet_x0_25) 読み込み中…")
    reid_ext = FastReIDExtractor(model_path="osnet_x0_25", device=args.device, use_onnx=False)
    reid_ext.load_model()
    print("  → 準備完了\n")

    # ── Step1: 疑似GT生成（original画質） ─────────────────────────────────────
    print("─── Step1: 疑似GT生成（original画質で実行） ───")
    t0 = time.perf_counter()
    pseudo_gt_tracks, _, _ = run_tracking(frames, "original", detector, reid_ext,
                                          args.gt_persons)
    print(f"  疑似GT生成完了 ({time.perf_counter()-t0:.0f}s)")
    print(f"  GTトラック数: {len({tid for fdata in pseudo_gt_tracks for tid,_ in fdata})}個\n")

    # ── Step2: 全画質で実行・評価 ─────────────────────────────────────────────
    sep = "─" * 105
    print(f"─── Step2: 画質比較 (GT={args.gt_persons}人, 設定=ReID-AGGR) ───\n")
    print(f"{'画質':<14} {'GTscore':>8} {'IDF1':>7} {'HOTA':>7} "
          f"{'DetA':>7} {'AssA':>7} {'frags':>6} {'Cover':>6} {'時間':>5}")
    print(sep)

    all_results: list[dict] = []

    for qcfg in QUALITY_PROFILES:
        print(f"  {qcfg['label']:<14}", end="  ", flush=True)
        t0 = time.perf_counter()

        try:
            # トラッキング実行
            pred_tracks, raw_m, det_times = run_tracking(
                frames, qcfg["key"], detector, reid_ext, args.gt_persons)

            # IDF1 + HOTA 計算
            evaluator = MOTEvaluator(iou_thresh=0.5)
            for gt_f, pred_f in zip(pseudo_gt_tracks, pred_tracks):
                evaluator.update(gt_f, pred_f)
            mot_m = evaluator.compute()

            elapsed = time.perf_counter() - t0
            m = {**raw_m, **mot_m, "key": qcfg["key"], "label": qcfg["label"],
                 "desc": qcfg["desc"], "total_s": elapsed}
            all_results.append(m)

            print(f"✓  {m['gt_score']:.3f}  {m['idf1']:.3f}  {m['hota']:.3f}  "
                  f"{m['deta']:.3f}  {m['assa']:.3f}  {m['frags']:>5}  "
                  f"{m['coverage']:.3f}  {elapsed:.0f}s")

            # 動画書き出し
            out_path = out_dir / f"{stem}_{qcfg['key']}.mp4"
            write_video(frames, pred_tracks, qcfg, out_path,
                        src_fps, args.skip, m, args.gt_persons, det_times)

        except Exception as e:
            import traceback
            elapsed = time.perf_counter() - t0
            print(f"✗  ERROR: {e}  ({elapsed:.0f}s)")
            traceback.print_exc()

    # ── サマリー動画 ──────────────────────────────────────────────────────────
    print(f"\nサマリー動画生成中…")
    try:
        sp = make_summary(out_dir, stem, src_fps, args.skip, all_results)
        print(f"  → {sp}")
    except Exception as e:
        print(f"  エラー: {e}")

    # ── 最終比較表 ────────────────────────────────────────────────────────────
    base = next((r for r in all_results if r["key"] == "original"), None)
    best_gts  = max((r["gt_score"] for r in all_results), default=0)
    best_idf1 = max((r["idf1"]     for r in all_results), default=0)
    best_hota = max((r["hota"]     for r in all_results), default=0)

    print(f"\n{sep}")
    print("  画質劣化による精度変化（ベースライン: ORIGINAL）\n")
    print(f"  {'画質':<14} │ {'GTscore':>8} {'Δ':>6} │ {'IDF1':>7} {'Δ':>6} │ "
          f"{'HOTA':>7} {'Δ':>6} │ {'frags':>5} │ {'DetA':>6} │ {'AssA':>6}")
    print(f"  {'─'*95}")
    for m in all_results:
        dg = m["gt_score"] - (base["gt_score"] if base else 0)
        di = m["idf1"]     - (base["idf1"]     if base else 0)
        dh = m["hota"]     - (base["hota"]     if base else 0)
        sg = "★" if m["gt_score"] >= best_gts  else " "
        si = "★" if m["idf1"]     >= best_idf1 else " "
        sh = "★" if m["hota"]     >= best_hota else " "
        print(f"  {m['label']:<14} │ {m['gt_score']:>8.3f} {dg:>+6.3f} │ "
              f"{m['idf1']:>6.3f}{si} {di:>+6.3f} │ "
              f"{m['hota']:>6.3f}{sh} {dh:>+6.3f} │ "
              f"{m['frags']:>5} │ {m['deta']:>6.3f} │ {m['assa']:>6.3f}")
    print(f"  {'─'*95}")

    print("""
  指標説明:
    GTscore  : 独自指標 — ID精度×Coverage×ODrateの調和平均
    IDF1     : Identification F1 — IDの一致率（高いほど正しいIDで追跡）
    HOTA     : Higher Order Tracking Accuracy — 検出+関連付けの複合指標
    DetA     : HOTAの検出精度コンポーネント（IoU≥0.05〜0.95の平均）
    AssA     : HOTAの関連付け精度コンポーネント（同一人物を一貫して追跡）
    ★        : 各指標で最高スコア
""")

    json_path = out_dir / "eval_quality2.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"  動画 → {out_dir}/  ({len(all_results)}本 + サマリー1本)")
    print(f"  JSON → {json_path}")


if __name__ == "__main__":
    main()
