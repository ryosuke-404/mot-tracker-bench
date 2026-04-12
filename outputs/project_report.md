# Multi-Object Tracking プロジェクト 実装レポート

> 作成日: 2026-04-12  
> 対象ディレクトリ: `/Users/smri/Downloads/cursor/ste`

---

## 1. プロジェクト概要

映像内の人物を複数同時に追跡（Multi-Object Tracking）し、
- どのトラッカーが最も安定してIDを維持できるか
- Re-IDモデルによってID消失からの回復（Rescue）がどう改善されるか

を定量評価・可視化することを目的としたプロジェクト。

---

## 2. 使用した技術スタック

| カテゴリ | 技術/ライブラリ |
|----------|----------------|
| 検出器 | YOLOX（yolox_s） |
| トラッキング | ByteTrack, BotSORT, StrongSORT, DeepSORT, HybridSORT, GHOST, OC-SORT 他 計23種 |
| Re-ID（軽量） | torchreid (osnet_x0_25 〜 osnet_ain_x1_0) |
| Re-ID（高精度） | boxmot（Market-1501 / MSMT17 / DukeMTMC 学習済み 37モデル） |
| 評価指標 | GTscore（独自）、IDprecision、Coverage、ODrate、frags |
| 可視化 | OpenCV (avc1/H.264) |

---

## 3. 実装したスクリプト一覧

| スクリプト | 内容 |
|-----------|------|
| `scripts/run_video.py` | 単一トラッカーで動画処理・可視化 |
| `scripts/run_video_compare.py` | 複数トラッカーを並べて比較動画生成 |
| `scripts/run_video_all.py` | 23トラッカー × 4後処理（none/ABC/DE/ABCDE）= 92動画を一括生成 |
| `scripts/run_reid_compare.py` | hybridsort_tuned × 37 Re-IDモデルを一括実行・評価 |
| `scripts/run_reid_rescue_vis.py` | Re-IDリカバリー発生を動画内にリアルタイム可視化 |
| `scripts/evaluate.py` | 動画に対してトラッカーを実行しGTscoreを計算 |

---

## 4. 評価指標（GTscore）の定義

```
IDprecision = GT人数 / 総ID生成数  （少ないほど良い = ID分裂が少ない）
Coverage    = 長期追跡ID数 / GT人数  （高いほど良い = 全員を継続追跡）
ODrate      = 平均同時検出数 / GT人数  （1.0に近いほど良い）
under_p     = min(1 / ODrate, 1.0)

GTscore = 調和平均(IDprecision, Coverage, under_p)
```

---

## 5. 使用した動画と結果

### test3.mp4（6人ダンス、カメラ前を通り過ぎる人あり）
- **GT人数**: 6人
- **主な課題**: オクルージョン多発、IDが頻繁に消えて再生成される

#### 23トラッカー × none の結果（上位5）

| 順位 | トラッカー | GTscore | frags |
|------|----------|---------|-------|
| 1 | hybridsort_tuned | 最高 | 最少 |
| 2〜 | botsort_tuned, bytetrack_tuned 他 | — | — |

### test4.mp4（2人すれ違い）
- **GT人数**: 2人
- **主な課題**: すれ違い時にIDが入れ替わる（ID Switch）
- **発見**: GTscore=1.000, frags=2 でも目視するとIDが逆になっていた
- **結論**: 現行の評価指標（GTscore）ではID Switchを検出できない

### test5.mp4（2人、Re-ID強化検証用）
- Re-IDを osnet_x0_25 → osnet_x1_0 に強化し DE後処理を組み合わせることで
  外観ベースのマージが機能し、ID Switch が改善

---

## 6. 後処理（Post-processing）の種類と効果

| 後処理 | 内容 | 効果 |
|--------|------|------|
| **none** | 後処理なし | ベースライン |
| **ABC** | オンライン：ロスト直後のTrackをリベラルに救済 | 短期消失に強い |
| **DE** | オフライン：外観類似度で同一人物のIDを統合 | ID Switchの修正に有効 |
| **ABCDE** | ABC + DE の組み合わせ | 総合的に最強だが過マッチのリスク |

**実装した2パス方式：**
- 1回目（Standard pass）: thresh=0.72, pruning有効 → none/DE用
- 2回目（Liberal pass）: thresh=0.60, pruning無効, hits≤5でrescue → ABC/ABCDE用

---

## 7. 23トラッカーの特性まとめ

| トラッカー | 特徴 | Re-ID対応 |
|-----------|------|----------|
| **hybridsort_tuned** | IoU×0.40 + height×0.20 + shape×0.10 + ReID×0.30 の4要素コスト。最も安定 | ○ |
| ByteTrack | 高速、シンプル、Low-score detectionを活用 | △（外観なし） |
| BotSORT | ByteTrack + カメラ補正 + Re-ID | ○ |
| StrongSORT | DeepSORT改良版、外観特徴重視 | ○ |
| DeepSORT | Re-ID特徴 + カルマンフィルタ | ○ |
| OC-SORT | 再検出時のコスト改善版SORT | △ |
| GHOST | 短期/長期の2段階マッチング | ○ |
| その他16種 | 各種チューニング済みバリアント | — |

---

## 8. Re-IDモデル比較（37モデル × hybridsort_tuned × test3.mp4）

### 上位モデル（frags=135 グループ）

| モデル | GTscore | frags | Rescue数 | 処理時間 |
|--------|---------|-------|---------|---------|
| **osnet_ibn_x1_0_msmt17** | 0.1224 | **135** | **53**（最少） | 2557s |
| hacnn_msmt17 | 0.1224 | 135 | 65 | **626s**（最速） |
| hacnn_market1501 | 0.1224 | 135 | 85 | 639s |
| osnet_x1_0_msmt17 | 0.1224 | 135 | 77 | 1124s |
| mobilenetv2_x1_4_msmt17 | 0.1224 | 135 | 99 | 1434s |
| osnet_ain_x1_0_msmt17 | 0.1224 | 135 | 73 | 1310s |

### 問題のあったモデル

| モデル | 問題 |
|--------|------|
| **lmbn_n_duke** | Coverage=0.167（6人中1人しか追跡できず）、処理時間6748秒（他の10倍） |
| **lmbn_n_market/cuhk03** | avg_rescue_sim=0.991（異常に高い）= ほぼ全員を同一人物と誤認する過剰マッチ |

---

## 9. 分かったこと・知見

### 9-1. ID分裂の根本原因
- test3.mp4のような**オクルージョン多発環境**では、どのモデルを使っても1人あたり平均20個以上のIDが生成される
- Re-IDによるRescueで100〜200回の統合は行われるが、それでもfrags=135（GT=6に対して22倍）
- これはRe-IDの性能問題というよりも**検出精度・追跡バッファの設定問題**の可能性が高い

### 9-2. Re-IDモデルの選び方
- **精度重視**: `osnet_ibn_x1_0_msmt17`  
  → Rescue=53（最少）= 本当に必要な時だけ発動し、誤マッチが最も少ない
- **速度重視**: `hacnn_msmt17`  
  → 626秒で同等の精度。実時間近くで処理したい場合に最適
- **避けるべき**: `lmbn_n_duke`（Coverage崩壊）、`lmbn_n_market/cuhk03`（過剰マッチ）
- **学習データセット傾向**: msmt17学習モデルは全体的にrescue数が少なく安定（汎化性能が高い）

### 9-3. GTscoreの限界
- **ID Switchを検出できない**: すれ違い後にIDが入れ替わっても、frags・coverageの値は変わらない
- 真の評価には MOTA/HOTA などの専用MOT評価指標が必要

### 9-4. 後処理の使い分け
- **短期オクルージョン（人が一瞬隠れる）** → ABC が効果的
- **長期消失後の再登場（人が画面外へ出て戻る）** → DE が効果的
- **ダンス動画のような複雑シーン** → ABCDEでも限界あり、Re-IDモデルの質が重要

### 9-5. 技術的な落とし穴

#### NumPy ABI 競合問題
```
症状: import torch が RuntimeError: Numpy is not available で失敗
原因: torch が NumPy 1.x でコンパイルされているが NumPy 2.x がインストールされている
解決: pip install "numpy<2.0" でダウングレード
```

#### macOS でのMP4書き出し問題
```
症状: VideoWriter が静かに失敗し、0バイトのMP4が生成される
原因: mp4v コーデックは macOS では動作しない
解決: avc1 (H.264) コーデックを使用する
  fourcc = cv2.VideoWriter_fourcc(*"avc1")
```

#### boxmot のモジュール読み込み順序
```
症状: import boxmot が早すぎると NumPy 初期化前にクラッシュ
解決: 遅延インポート（lazy import）パターンを使用
  def _get_boxmot():
      import torch
      import boxmot
      from boxmot.reid.core.auto_backend import ReidAutoBackend
      return ReidAutoBackend
```

---

## 10. 推奨構成（まとめ）

### 一般的なシーン（人物追跡）
```
トラッカー : hybridsort_tuned
Re-ID     : osnet_ibn_x1_0_msmt17（精度優先）
            hacnn_msmt17（速度優先）
後処理     : DE（長期追跡が重要な場合）
```

### リアルタイム処理が必要な場合
```
トラッカー : bytetrack_tuned または botsort_tuned
Re-ID     : hacnn_msmt17（最速）
後処理     : ABC（オンラインのみ）
```

### ID Switch が問題になる場合（すれ違い多発）
```
追加対策:
  - osnet_x1_0 以上の高精度Re-IDを使用
  - DE後処理で外観マージを有効化
  - track_buffer を長く設定（180フレーム以上）
  - 評価にはMOTA/HOTAを使用（GTscoreでは検出不可）
```

---

## 11. 生成した主な出力ファイル

| パス | 内容 |
|------|------|
| `outputs/reid_compare/eval_reid_compare.json` | 37 Re-IDモデルの全評価結果 |
| `outputs/reid_compare/*.mp4` | 37モデル × hybridsort_tuned の追跡動画 |
| `outputs/reid_rescue_vis/test3_hybridsort_tuned_osnet_ibn_x1_0_msmt17_rescue.mp4` | Re-IDリカバリーを可視化した動画（53回のrescueを画面表示）|
| `outputs/reid_rescue_vis/test3_..._result.json` | rescue発生フレーム・ID・類似度の詳細ログ |
| `outputs/compare/*.mp4` | トラッカー比較動画（test.mp4）|
| `outputs/test4_combos/*.mp4` | test4.mp4（すれ違い）× 23トラッカー |
| `outputs/test5_combos/*.mp4` | test5.mp4 × 全Re-IDモデル × ABC/DE |
