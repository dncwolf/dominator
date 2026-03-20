"""PSYCHO-PASS 犯罪係数解析システム - YouTube URL エントリポイント"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import cv2
import yt_dlp

from psycho_pass.analyze import analyze


def download(url: str, out_dir: str) -> tuple[str, str]:
    """yt-dlp で mp4 保存。(動画ファイルパス, video_id) を返す。"""
    os.makedirs(out_dir, exist_ok=True)

    ydl_opts = {
        "format": "best[ext=mp4]/best",
        "outtmpl": os.path.join(out_dir, "%(id)s.%(ext)s"),
        "quiet": False,
        "no_warnings": False,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)

    video_id: str = info["id"]
    ext = info.get("ext", "mp4")
    video_path = os.path.join(out_dir, f"{video_id}.{ext}")

    if not os.path.exists(video_path):
        # 拡張子が違う場合も探す
        for f in Path(out_dir).glob(f"{video_id}.*"):
            if f.suffix.lower() in (".mp4", ".mkv", ".webm", ".avi"):
                video_path = str(f)
                break

    return (video_path, video_id)


def detect_scenes(video_path: str, threshold: float = 30.0) -> list[tuple[float, float]]:
    """フレーム間差分でシーン区間 (start_sec, end_sec) のリストを返す。"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"動画を開けませんでした: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_sec = total_frames / fps

    scenes: list[tuple[float, float]] = []
    scene_start_frame = 0
    prev_gray = None
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if prev_gray is not None:
            diff = cv2.absdiff(prev_gray, gray)
            mean_diff = float(diff.mean())

            if mean_diff > threshold:
                start_sec = scene_start_frame / fps
                end_sec = frame_idx / fps
                if end_sec > start_sec:
                    scenes.append((start_sec, end_sec))
                scene_start_frame = frame_idx

        prev_gray = gray
        frame_idx += 1

    start_sec = scene_start_frame / fps
    end_sec = total_sec
    if end_sec > start_sec:
        scenes.append((start_sec, end_sec))

    cap.release()

    if not scenes:
        scenes = [(0.0, total_sec)]

    return scenes


def extract_best_frame(
    video_path: str,
    scene_start: float,
    scene_end: float,
    out_dir: str,
    scene_idx: int = 0,
) -> str:
    """シーン区間内を最大 5 点サンプリングし、ラプラシアン分散が最大のフレームを保存。"""
    os.makedirs(out_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"動画を開けませんでした: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    duration = scene_end - scene_start

    n_samples = min(5, max(1, int(duration * fps)))
    if n_samples == 1:
        sample_times = [scene_start]
    else:
        step = duration / (n_samples - 1) if n_samples > 1 else duration
        sample_times = [scene_start + i * step for i in range(n_samples)]
        sample_times = [min(t, scene_end - 1.0 / fps) for t in sample_times]

    best_frame = None
    best_score = -1.0
    best_timestamp = scene_start

    for t in sample_times:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        if lap_var > best_score:
            best_score = lap_var
            best_frame = frame
            best_timestamp = t

    cap.release()

    if best_frame is None:
        raise RuntimeError(f"フレームを取得できませんでした: scene={scene_idx}")

    # ブレ判定: 100 未満でも最高スコアを採用（仕様通り）
    filename = f"scene_{scene_idx:03d}_t{best_timestamp:.2f}s.jpg"
    out_path = os.path.join(out_dir, filename)
    cv2.imwrite(out_path, best_frame)
    return out_path


def analyze_all(frame_paths: list[str], timestamps: list[float]) -> list[dict]:
    """各フレームに analyze() を適用。"""
    all_results: list[dict] = []

    for frame_path, timestamp in zip(frame_paths, timestamps):
        print(f"\n[analyze_all] フレーム解析: {frame_path} (t={timestamp:.2f}s)")
        try:
            persons = analyze(frame_path)
        except Exception as exc:
            print(f"  [警告] 解析失敗: {exc}")
            persons = []

        persons_summary = [
            {
                "id": p["id"],
                "coefficient": p["coefficient"],
                "label": p["label"],
            }
            for p in persons
        ]

        all_results.append({
            "timestamp": timestamp,
            "frame": frame_path,
            "persons": persons_summary,
        })

    return all_results


def save_summary(results: list[dict], video_out_dir: str) -> str:
    """{video_out_dir}/summary.csv を保存してパスを返す。"""
    os.makedirs(video_out_dir, exist_ok=True)
    csv_path = os.path.join(video_out_dir, "summary.csv")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["scene", "timestamp", "person_id", "coefficient", "label"])

        for scene_idx, scene_result in enumerate(results):
            timestamp = scene_result["timestamp"]
            persons = scene_result["persons"]
            if not persons:
                writer.writerow([scene_idx, f"{timestamp:.2f}", "", "", "人物未検出"])
            else:
                for person in persons:
                    writer.writerow([
                        scene_idx,
                        f"{timestamp:.2f}",
                        person["id"],
                        person["coefficient"],
                        person["label"],
                    ])

    print(f"\n[save_summary] CSV 保存: {csv_path}")
    return csv_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PSYCHO-PASS 犯罪係数解析システム",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  python -m psycho_pass.dominator https://youtube.com/watch?v=xxx
  python -m psycho_pass.dominator https://youtube.com/watch?v=xxx --threshold 25.0 --out ./result
        """,
    )
    parser.add_argument("url", help="解析する YouTube URL")
    parser.add_argument(
        "--threshold",
        type=float,
        default=30.0,
        help="シーン検知の差分しきい値 (デフォルト: 30.0, 低い=細かく/高い=大きな変化のみ)",
    )
    parser.add_argument(
        "--out",
        default="./output",
        help="出力ディレクトリ (デフォルト: ./output)",
    )
    parser.add_argument(
        "--max-scenes",
        type=int,
        default=10,
        help="抽出する最大シーン数 (デフォルト: 10)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("PSYCHO-PASS 犯罪係数解析システム")
    print("=" * 60)
    print(f"URL: {args.url}")
    print(f"シーン検知しきい値: {args.threshold}")
    print(f"最大シーン数: {args.max_scenes}")
    print(f"出力先: {args.out}")
    print()

    print("[Step 1] 動画ダウンロード中...")
    video_path, video_id = download(args.url, args.out)
    print(f"  動画 ID: {video_id}")
    print(f"  保存先: {video_path}")

    video_out_dir = os.path.join(args.out, video_id)
    frames_dir = os.path.join(video_out_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    print(f"\n[Step 2] シーン検知中 (しきい値={args.threshold})...")
    scenes = detect_scenes(video_path, threshold=args.threshold)
    print(f"  検出シーン数: {len(scenes)}")
    if len(scenes) > args.max_scenes:
        # 動画全体から均等にサンプリング
        step = len(scenes) / args.max_scenes
        scenes = [scenes[int(i * step)] for i in range(args.max_scenes)]
        print(f"  -> {args.max_scenes} シーンに間引き")

    print(f"\n[Step 3] ベストフレーム抽出中...")
    frame_paths: list[str] = []
    timestamps: list[float] = []

    for scene_idx, (start_sec, end_sec) in enumerate(scenes):
        try:
            frame_path = extract_best_frame(
                video_path, start_sec, end_sec, frames_dir, scene_idx=scene_idx
            )
            timestamp = (start_sec + end_sec) / 2.0
            frame_paths.append(frame_path)
            timestamps.append(timestamp)
            print(f"  Scene {scene_idx:03d}: {start_sec:.2f}s ~ {end_sec:.2f}s -> {os.path.basename(frame_path)}")
        except Exception as exc:
            print(f"  [警告] Scene {scene_idx} のフレーム抽出失敗: {exc}")

    print(f"\n[Step 4] 犯罪係数解析中...")
    results = analyze_all(frame_paths, timestamps)

    print(f"\n[Step 5] サマリー保存中...")
    csv_path = save_summary(results, video_out_dir)

    print("\n" + "=" * 60)
    print("解析完了")
    print("=" * 60)
    total_persons = sum(len(r["persons"]) for r in results)
    print(f"  解析シーン数: {len(results)}")
    print(f"  検出人物数合計: {total_persons}")
    print(f"  サマリー CSV: {csv_path}")
    print(f"  フレーム保存先: {frames_dir}")

    print("\n--- 要注意人物 ---")
    any_threat = False
    for scene_idx, scene_result in enumerate(results):
        for person in scene_result["persons"]:
            if person["label"] != "執行対象外（クリア）":
                print(
                    f"  Scene {scene_idx:03d} (t={scene_result['timestamp']:.2f}s)"
                    f" Person {person['id']}: 係数={person['coefficient']} [{person['label']}]"
                )
                any_threat = True
    if not any_threat:
        print("  なし（全員クリア）")


if __name__ == "__main__":
    main()
