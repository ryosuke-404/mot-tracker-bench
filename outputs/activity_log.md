# 作業ログ — 日時別

---

## Apr 9 (水)

### 01:46 〜 02:24
- `run_video.py` で最初の動画処理テスト (`result.json`, `result_v2.json`)
- ByteTrack + ResNet50 Re-ID の基本動作確認

### 02:24 〜 11:03
- `result_v3.json`, `result_yolox.json` 生成
- YOLOX 検出器の統合・動作確認

### 11:03 〜 13:06
- `eval_results.json` 生成 (初期ベースライン評価)
- 複数検出器 × トラッカーの組み合わせ評価スクリプト (`evaluate.py`) の骨格構築
- GTscore メトリクス設計・実装

---

## Apr 10 (木)

### 17:20
- `eval_all_trackers.json` 生成
- 15種類の全トラッカーを一括評価
  - ByteTrack, SORT, OC-SORT, DeepSORT, StrongSORT, FairMOT, CB-IoU, HybridSORT, UCMCTrack, DeepOC-SORT, SMILETrack, SparseTrack, GHOST, TransTrack, BoT-SORT
- Re-IDギャラリー統合、rescue カウント追加

### 19:15
- `eval_gt7.json` 生成
- GT=7人として GT-based メトリクス (GTscore, IDprecision, Coverage, ODrate) 本格導入
- 全15トラッカーの GT ベース精度ランキング確定

### 21:17
- `eval_tuned.json` 生成 — Round 1 パラメータチューニング
  - SORT: track_thresh 0.45→0.60 → ODrate 1.97x→1.35x, frags 26→16
  - BoT-SORT: gmc_method "orb"→"none" (CPU不安定性解消)
  - SparseTrack: n_layers 3→1

### 21:39
- `eval_tuned2.json` 生成 — Round 2 パラメータチューニング
  - 断片化型トラッカー (ByteTrack, OC-SORT, HybridSORT, DeepOC-SORT, TransTrack) に共通チューニング
  - track_buffer 90→180, match_thresh 0.85→0.90
  - sort_tuned (GTscore=0.647, frags=16) が全体1位に

### 21:52
- `evaluation_report.md` 更新
  - 21バリアント対応の詳細ランキング表
  - sort_tuned を本番推奨として明記

### 21:52 〜 22:30 (推定)
- Re-IDの効果分析
  - rescue件数 vs frags の関係を整理
  - Re-IDは断片化を完全には解消できないことを確認
- 同一ID維持戦略のアイデア検討 (戦略A〜E)
- 戦略D・E の設計
  - D: Union-Find による Appearance merge
  - E: AgglomerativeClustering による k=gt_persons 強制集約

### 22:30 〜 23:28 (推定)
- `tracking/postprocess.py` 新規実装
  - `compute_mean_embeddings()`
  - `merge_by_appearance()` — 戦略D
  - `cluster_reassign()` — 戦略E (scikit-learn AgglomerativeClustering)
  - `apply_id_map()`, `compose_maps()`
- `evaluate.py` に `--postprocess` フラグ追加
- DE後処理の初回評価実行 → `eval_postprocess.json`
  - ocsort_tuned, hybridsort_tuned, bytetrack が 0.284→0.882 に大幅改善

### 23:28 〜 00:00
- scikit-learn 未インストールエラー発生 → `.venv/bin/pip install scikit-learn` で解消
- ABC戦略 (戦略A+B+C) の実装
  - liberal_rescue モード: thresh 0.72→0.60, gallery 無制限, rescue hits≤5
- `eval_abc.json` 生成 (7コンボ, ABCのみ)
  - ABC単体では精度向上なし (sort_tuned=0.647 が依然最高)

---

## Apr 11 (金)

### 00:41 〜 00:45
- `eval_abc.json` 完成 (ABCのみ評価)
- `eval_abcde.json` 生成 (A〜E全組み合わせ)
  - ABCDE: DEより悪化 (ABCが埋め込み品質を劣化させDE精度を下げる)
- 精度のみでのランキング整理:
  - 最高: ocsort_tuned/hybridsort_tuned/bytetrack + DE後処理 = **0.882**
  - リアルタイム最高: sort_tuned = **0.647**

### 01:35 〜 01:41
- `eval_res075.json`, `eval_res050.json`, `eval_res025.json` 生成
- 解像度影響調査 (degrade_res パラメータ実装)
  - 100% (2560×1440) → 75% → 50% → 25% (640×360)
  - 全解像度で GTscore ほぼ同値
  - 結論: この映像では解像度を25%まで落とせる

### 02:20 〜 03:03
- `scripts/run_video_compare.py` 新規作成
  - 5トラッカー × test.mp4 → 個別アノテーション動画を生成
  - 各動画にトラッカー名の大きなカラーバナーを表示 (色分け識別)
- 5本の比較動画を `outputs/compare/` に出力:
  - `test_sort_tuned.mp4` (シアン)
  - `test_ocsort_tuned.mp4` (ライムグリーン)
  - `test_bytetrack.mp4` (ブルー)
  - `test_deepocsort_tuned.mp4` (マゼンタ)
  - `test_hybridsort_tuned.mp4` (オレンジ)

### 03:01
- `eval_test.json` 生成 — test.mp4 の精度評価
  - sort_tuned が GTscore=0.300 で1位 (frags=51)
  - ocsort/deepocsort/hybridsort が GTscore=0.228 (frags=78)
  - bytetrack が GTscore=0.183 (frags=101)
  - 学習用動画より難易度が高い映像 (最高 0.300 vs 0.647)

---

## 成果物一覧

| ファイル | 生成日時 | 内容 |
|---------|---------|------|
| `outputs/eval_all_trackers.json` | Apr 10 17:20 | 全15トラッカー評価 |
| `outputs/eval_gt7.json` | Apr 10 19:15 | GT=7 GTscore評価 |
| `outputs/eval_tuned.json` | Apr 10 21:17 | Round1チューニング |
| `outputs/eval_tuned2.json` | Apr 10 21:39 | Round2チューニング |
| `outputs/evaluation_report.md` | Apr 10 21:52 | 詳細評価レポート |
| `outputs/eval_postprocess.json` | Apr 10 23:59 | DE後処理評価 |
| `outputs/eval_abc.json` | Apr 11 00:41 | ABC戦略評価 |
| `outputs/eval_abcde.json` | Apr 11 00:45 | ABCDE全戦略評価 |
| `outputs/eval_res075.json` | Apr 11 01:35 | 解像度75%評価 |
| `outputs/eval_res050.json` | Apr 11 01:37 | 解像度50%評価 |
| `outputs/eval_res025.json` | Apr 11 01:41 | 解像度25%評価 |
| `outputs/eval_test.json` | Apr 11 03:01 | test.mp4 精度評価 |
| `outputs/compare/test_*.mp4` | Apr 11 02:29〜03:03 | 5本の比較動画 |
| `outputs/project_summary.md` | Apr 11 (今回) | 本プロジェクト総括 |
| `outputs/activity_log.md` | Apr 11 (今回) | 本作業ログ |
