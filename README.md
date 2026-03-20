# DOMINATOR

YouTube 動画から人物を検出し、感情・姿勢・文脈情報をもとに犯罪係数を算出するローカル実行ツール。

## 概要

```
YouTube URL → yt-dlp → シーン検知 → フレーム抽出 → 解析 → summary.csv
```

- **感情解析**: DeepFace で表情から emotion_score を算出
- **姿勢解析**: YOLOv8-pose の骨格推定から posture_score を算出
- **文脈解析**: YOLOv8x で危険物（銃・刃物など）を検出し context_score を算出

## 犯罪係数の区分

| 係数 | 判定 | ドミネーター |
|---|---|---|
| 0〜99 | クリア | トリガーロック |
| 100〜299 | 潜在犯 | ノンリーサル・パラライザー |
| 300〜 | 執行対象 | リーサル・エリミネーター |
| 測定不能 | 免罪体質/機械等 | デストロイ・デコンポーザー |

## インストール

```bash
uv add ultralytics deepface tf-keras opencv-python yt-dlp pillow
```

## 使い方

```bash
python main.py <URL> [--threshold 30.0] [--out ./output] [--max-scenes 10]
```

```bash
python main.py https://youtube.com/watch?v=xxx
python main.py https://youtube.com/watch?v=xxx --threshold 25.0 --out ./result
python main.py https://youtube.com/watch?v=xxx --max-scenes 5
```

## 出力

```
output/
  └── {video_id}/
        ├── frames/
        │   ├── scene_000_t0.00s.jpg
        │   └── ...
        └── summary.csv   # scene, timestamp, person_id, coefficient, label
```

## 注意事項

- yt-dlp の使用はローカル・研究用途に限定
- DeepFace は初回実行時に重みを自動ダウンロード（時間がかかる）
- 長尺動画は `--threshold 40〜50` でフレーム数を削減することを推奨
