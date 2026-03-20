# PSYCHO-PASS 犯罪係数解析システム 仕様書 v1.0

## 1. 概要

YouTube URL を入力として受け取り、画像内の人物を検出、複数の視覚的手がかりから犯罪係数を算出し、原作準拠の区分と執行モードで判定結果を出力するローカル実行システム。

## 2. システム構成

**YouTube URL の処理フロー**

```
YouTube URL
  └─ yt-dlp           動画ダウンロード（mp4）
      └─ OpenCV       フレーム間差分によるシーン検知
          └─ 各シーン代表フレームを抽出
              └─ 解析パイプライン
                  └─ 画像保存 + summary.csv
```

## 3. 依存ライブラリ

| ライブラリ | 用途 | 備考 |
|---|---|---|
| ultralytics | 人物検出 / 姿勢推定 | yolov8x.pt, yolov8x-pose.pt |
| deepface | 感情・表情・年齢・性別解析 | tf-keras が必要 |
| opencv-python | シーン検知・フレーム抽出・差分計算 | |
| yt-dlp | YouTube 動画ダウンロード | ローカル・研究用途限定 |
| pillow | 描画・画像保存 | |

**インストール:**

```
uv add ultralytics deepface tf-keras opencv-python yt-dlp pillow
```

## 4. ファイル構成

```
psycho_pass/
  ├── analyze.py              共通解析ロジック（検出・スコアリング・描画）
  └── dominator.py  YouTube URL エントリポイント
```

## 5. 犯罪係数の仕様（原作準拠）

### 区分と執行モード

| 係数 | 区分 | 執行モード | ドミネーター状態 |
|---|---|---|---|
| 0 〜 99 | 執行対象外（クリア） | — | トリガーロック |
| 100 〜 299 | 潜在犯 / 要制圧 | ノンリーサル・パラライザー | 非殺傷麻痺モード |
| 300 〜 | 執行対象 / 即時排除 | リーサル・エリミネーター | 殺傷モード |
| 測定不能 | 免罪体質 / 機械等 | デストロイ・デコンポーザー | 最大破壊モード |

## 6. スコアリングエンジン

### 算出式

```
raw_score   = emotion_score * 0.40
            + posture_score * 0.35
            + context_score * 0.25

coefficient = int(raw_score * 400)   # 通常人間の上限: ~400
```

> **測定不能の条件:** `raw_score > 1.0`（複数の危険物検出 + 高感情スコアが重なった場合）

### 各スコアの算出方法

**emotion_score（DeepFace）**

感情ごとの重み付き和を 100 で正規化し、`[0.0, 1.0]` にクランプ

```
weighted_sum = Σ (emotion_prob[e] * weight[e])   # emotion_prob は 0〜100、合計≒100
emotion_score = clip(weighted_sum / 100, 0.0, 1.0)
```

| 感情 | 重み |
|---|---|
| angry | 1.0 |
| fear | 0.8 |
| disgust | 0.7 |
| sad | 0.4 |
| surprise | 0.3 |
| neutral | 0.1 |
| happy | -0.2 |

- 顔未検出（後ろ向き・遠距離）: デフォルト **0.15**（クリア寄り）
- DeepFace 例外（検出試行は成功したが感情取得失敗）: デフォルト **0.3**

**posture_score（YOLOv8-pose 骨格）**

信頼度 0.5 以上のキーポイントのみ使用。

| 指標 | 算出方法 | 寄与 |
|---|---|---|
| 肘角度（左右平均） | `arccos` で関節角を算出。90°以下で最大、180°で最小 | `(180 - elbow_deg) / 180` |
| 肩の非対称性 | 左右肩 y 座標差 / 人物クロップ高さ | 差が大きいほど高スコア（上限 0.3） |
| 頭部前傾 | 鼻 x が肩中点 x から ±10% 以上ズレ | +0.15 |

```
posture_score = clip(elbow_tension * 0.6 + shoulder_asym * 0.4 + head_lean, 0.0, 1.0)
```

- キーポイント取得数が 5 未満: デフォルト **0.2**（信頼できる推定不可）

**context_score（YOLOv8x 全物体検出）**

- confidence ≥ 0.5 の検出のみ対象
- 人物クロップの周囲 1.5 倍領域内に存在するオブジェクトのみ加点

| クラス | 加点 |
|---|---|
| gun | +0.40 |
| knife | +0.25 |
| baseball bat | +0.20 |
| scissors | +0.15 |
| bottle（割れ物判定: 信頼度 ≥ 0.7 かつ人物と重なる）| +0.10 |

```
context_score = clip(Σ 加点, 0.0, 1.0)
```

## 7. 関数 API

### analyze.py

```python
detect_persons(image_path: str) -> list[Box]
    # YOLOv8x で人物のみ検出（confidence >= 0.5）

emotion_score(face_crop: ndarray) -> float
    # DeepFace で感情スコア算出。clip(weighted_sum / 100, 0.0, 1.0)
    # 顔未検出: 0.15 / 感情取得失敗: 0.3

posture_score(keypoints, confidences) -> float
    # 骨格テンション算出。confidence >= 0.5 のキーポイントのみ使用
    # 有効キーポイント < 5: 0.2

context_score(person_box: Box, all_boxes: list[Box], image_shape: tuple[int, int]) -> float
    # 人物クロップ 1.5 倍領域内の危険物を confidence >= 0.5 で検出・加点
    # clip(合計, 0.0, 1.0)

classify(coeff: int, raw_score: float = 0.0) -> tuple[str, str]
    # 区分名と HEX カラーコードを返す
    # raw_score > 1.0 の場合は "測定不能" を返す

analyze(image_path: str) -> list[dict]
    # フルパイプライン実行・画像保存
```

### dominator.py

```python
download(url: str, out_dir: str) -> tuple[str, str]
    # yt-dlp で mp4 保存、(動画ファイルパス, video_id) を返す

detect_scenes(video_path: str, threshold: float = 30.0) -> list[tuple[float, float]]
    # フレーム差分でシーン区間 (start_sec, end_sec) のリストを返す
    # threshold の目安: 低い(15〜20)=細かく検知 / 高い(40〜50)=大きな変化のみ

extract_best_frame(video_path: str, scene_start: float, scene_end: float, out_dir: str, scene_idx: int = 0) -> str
    # シーン区間内を最大 5 点サンプリングし、ラプラシアン分散が最大の
    # （最も鮮鋭な）フレームを jpg 保存してパスを返す

analyze_all(frame_paths: list[str], timestamps: list[float]) -> list[dict]
    # 各フレームに analyze() を適用
    # 戻り値: [{"timestamp": float, "frame": str, "persons": [
    #            {"id": int, "coefficient": int, "label": str}
    #          ]}]

save_summary(results: list[dict], video_out_dir: str) -> str
    # {video_out_dir}/summary.csv を保存してパスを返す
```

## 8. CLI

```bash
python main.py <URL> [--threshold 30.0] [--out ./output] [--max-scenes 10]
```

**例:**

```bash
python main.py https://youtube.com/watch?v=xxx
python main.py https://youtube.com/watch?v=xxx --threshold 25.0 --out ./result
python main.py https://youtube.com/watch?v=xxx --max-scenes 5
```

## 9. 出力

動画ごとにサブディレクトリを作成する。ディレクトリ名は YouTube の動画 ID。

```
output/
  └── {video_id}/
        ├── frames/
        │   ├── scene_000_t0.00s.jpg
        │   ├── scene_001_t8.34s.jpg
        │   └── ...
        └── summary.csv
```

**summary.csv のカラム:**

```
scene, timestamp, person_id, coefficient, label
```

## 10. 制約・注意事項

- yt-dlp の使用はローカル・研究用途に限定すること
- DeepFace は初回実行時に重みを自動ダウンロードするため時間がかかる
- 長尺動画は `--threshold` を大きく（40〜50）設定してフレーム数を減らすことを推奨
- 人物が小さい・後ろ向きの場合、顔検出が失敗し emotion_score = 0.15（クリア寄り）になる
- 係数の最高値は原作で 832（東金朔夜）。通常人物の実用上限は ~400 を想定
- `extract_best_frame` のラプラシアン分散しきい値（ブレ判定）: 100 未満のフレームは全候補がブレていても最高スコアのものを採用する
