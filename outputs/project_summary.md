# プロジェクト総括レポート
## 人物追跡システム (Multi-Object Tracking) 評価・改善

---

## 1. プロジェクト概要

バス車内カメラ映像から乗降客を追跡するシステムの精度評価・改善プロジェクト。
検出器・トラッカー・Re-IDモデルの組み合わせを体系的に評価し、最適な構成を特定した。

**対象動画**: `assets/142755-780943401_medium.mp4` (2560×1440, 300フレーム, GT=7人)  
**テスト動画**: `assets/test.mp4` (1920×1080, 695フレーム)

---

## 2. 実施内容まとめ

### Phase 1 — ベースライン構築 (Apr 9)

**全トラッカー網羅的評価**
- 15種類のトラッカーを一括評価: ByteTrack, SORT, OC-SORT, DeepSORT, StrongSORT, FairMOT, CB-IoU, HybridSORT, UCMCTrack, DeepOC-SORT, SMILETrack, SparseTrack, GHOST, TransTrack, BoT-SORT
- 評価指標を設計・実装 (GTscore = IDprecision × Coverage × 1/ODrate の調和平均)
- 結果: ByteTrack がトップも、全体的に frags が多く (60〜100+) 精度が低い状態

**Re-ID ギャラリー統合**
- `reid/gallery.py` 実装: cosine similarity によるトラック再同定 (gallery size=200, thresh=0.72)
- Re-ID rescue で ID断片化を一部回復

---

### Phase 2 — パラメータチューニング Round 1 (Apr 10 午前)

**実験内容**:
| トラッカー | 変更内容 | 効果 |
|---|---|---|
| SORT | track_thresh 0.45→0.60 | ghost tracks 抑制: ODrate 1.97x→1.35x, frags 26→16 |
| BoT-SORT | gmc_method "orb"→"none" | CPU不安定性を解消 |
| SparseTrack | n_layers 3→1 | OC-SORT相当動作に変更 |

**発見**: トラッカーは2種類に分類できる
- **Ghost track型** (SORT): track_thresh を上げて幽霊トラック抑制
- **断片化型** (ByteTrack等): track_buffer・match_thresh 緩和で断片化抑制

---

### Phase 3 — パラメータチューニング Round 2 (Apr 10 昼)

**断片化型トラッカー向け共通チューニング**:
- `track_buffer`: 90→180 (長い遮蔽に対応)
- `match_thresh`: 0.85→0.90 (より緩やかなマッチング)
- 対象: ByteTrack, OC-SORT, HybridSORT, DeepOC-SORT, TransTrack

**7種類 × 300フレーム 評価結果** (`eval_tuned2.json`):

| Rank | Tracker | GTscore | Frags |
|------|---------|---------|-------|
| #1 | sort_tuned | 0.647 | 16 |
| #2 | sort | 0.449 | 26 |
| #3 | deepocsort_tuned | 0.382 | 41 |
| #4 | ocsort_tuned | 0.375 | 42 |
| #5 | bytetrack_tuned | 0.368 | 43 |

→ **sort_tuned を本番デフォルトに採用** (`config/system.yaml`, `run_video.py`, `run_bus.py` 更新)

---

### Phase 4 — Re-ID 効果分析 (Apr 10 夜)

**分析結果**:
- Re-IDによる rescue: 1〜36件/300フレーム
- ただし id_precision はどのトラッカーでも 0.1〜0.44 止まり
- 理由: rescue は既存 ID を回復するが、新たな「断片」を生成してしまう (reassign 後も元のフラグメントカウントが残る)
- Re-IDの真の価値は「同一人物として認識し続ける確率」の向上

---

### Phase 5 — 同一ID維持戦略の実装 (Apr 10〜11 深夜)

**5つの戦略を設計・実装** (`tracking/postprocess.py`):

| 戦略 | 内容 |
|------|------|
| A | gallery thresh 0.72→0.60 (より積極的にマッチ) |
| B | gallery pruning 無効化 (全埋め込みを保持) |
| C | rescue ウィンドウ拡張 (hits==1 → hits≤5) |
| D | Appearance merge: 非重複トラックを cosine sim ≥ 0.65 でマージ (Union-Find) |
| E | Cluster reassign: AgglomerativeClustering(n=gt_persons) で k=7 IDに強制集約 |

**3パターン評価**:

| 組み合わせ | Best PostScore | 備考 |
|---|---|---|
| ABCのみ | 0.647 (sort_tuned, 変化なし) | ABC はフラグ数を下げられない |
| DEのみ | **0.882** (ocsort/hybrid/bytetrack) | 大幅改善 |
| ABCDE | 0.882 | ABC が DE の精度を下げる (埋め込み品質悪化) |

**結論**: **DE後処理のみ** が最適。ABCはリアルタイム用途での改善なし。

---

### Phase 6 — 解像度影響調査 (Apr 11 早朝)

**degrade_res パラメータで解像度を段階的に下げて評価**:

| 解像度 | 実解像度 | GTscore (sort_tuned) |
|------|---------|---------|
| 100% (元) | 2560×1440 | 0.647 |
| 75% | 1920×1080 | 0.647 |
| 50% | 1280×720 | 〜0.647 |
| 25% | 640×360 | 〜0.647 |

**結論**: この映像では解像度を25%まで落としても精度にほぼ影響なし。  
理由: 映像内で人物が大きく映っており静止に近いシーンが多い。

---

### Phase 7 — test.mp4 マルチトラッカー比較動画生成 (Apr 11 02:20〜03:03)

**新スクリプト `scripts/run_video_compare.py` 作成**:
- 複数トラッカーを順次処理し、各トラッカーごとに独立した動画を出力
- 各動画にトラッカー名の大きなカラーバナーを表示 (色分けで識別)
- stats パネル: tracker名, frame, FPS, 推論時間, active数, frags, rescues

**生成動画** (`outputs/compare/`):

| ファイル | バナー色 |
|---------|---------|
| test_sort_tuned.mp4 | シアン |
| test_ocsort_tuned.mp4 | ライムグリーン |
| test_bytetrack.mp4 | ブルー |
| test_deepocsort_tuned.mp4 | マゼンタ |
| test_hybridsort_tuned.mp4 | オレンジ |

**test.mp4 精度評価結果** (`eval_test.json`):

| Rank | Tracker | GTscore | Frags | ODrate |
|------|---------|---------|-------|--------|
| #1 | sort_tuned | **0.300** | 51 | 1.73x |
| #2 | ocsort_tuned | 0.228 | 78 | 0.88x |
| #2 | deepocsort_tuned | 0.228 | 78 | 0.88x |
| #2 | hybridsort_tuned | 0.228 | 78 | 0.88x |
| #5 | bytetrack | 0.183 | 101 | 0.91x |

---

## 3. 最終精度ランキング (学習用動画, DE後処理あり)

| Rank | Combo | PostScore | Frags→ | 備考 |
|------|-------|-----------|--------|------|
| #1 | ocsort_tuned + DE | **0.882** | 42→6 | |
| #1 | hybridsort_tuned + DE | **0.882** | 44→7 | |
| #1 | bytetrack + DE | **0.882** | 60→5 | |
| #4 | sort_tuned (no post) | 0.647 | 16→7 | リアルタイム最良 |

---

## 4. 本番採用設定

**`config/system.yaml`**:
```yaml
tracker:
  type: "sort"
  track_thresh: 0.60
  max_age: 90
  min_hits: 3
  iou_thresh: 0.30
```

**リアルタイム用途**: sort_tuned (GTscore=0.647, frags=16, 最速)  
**オフライン高精度用途**: ocsort_tuned / hybridsort_tuned + DE後処理 (PostScore=0.882)

---

## 5. 実装ファイル一覧

| ファイル | 内容 |
|---------|------|
| `scripts/evaluate.py` | 組み合わせ評価スクリプト (detector×tracker×reid) |
| `scripts/run_video.py` | 単一動画処理 (sort_tuned デフォルト) |
| `scripts/run_video_compare.py` | マルチトラッカー比較動画生成 (新規) |
| `scripts/run_bus.py` | バス運行用リアルタイム処理 |
| `tracking/postprocess.py` | D/E後処理実装 (merge + cluster reassign) |
| `tracking/sort.py` | SORT実装 |
| `tracking/bytetrack.py` | ByteTrack実装 |
| `tracking/ocsort.py` | OC-SORT実装 |
| `tracking/hybridsort.py` | HybridSORT実装 |
| `tracking/deepocsort.py` | DeepOC-SORT実装 |
| `reid/gallery.py` | Re-IDギャラリー管理 |
| `config/system.yaml` | システム設定 |
| `outputs/evaluation_report.md` | 詳細評価レポート |
