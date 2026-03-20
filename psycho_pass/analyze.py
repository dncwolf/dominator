"""PSYCHO-PASS 犯罪係数解析モジュール"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

Box = dict  # {"x1": int, "y1": int, "x2": int, "y2": int, "conf": float, "class": str}

EMOTION_WEIGHTS: dict[str, float] = {
    "angry": 1.0,
    "fear": 0.8,
    "disgust": 0.7,
    "sad": 0.4,
    "surprise": 0.3,
    "neutral": 0.1,
    "happy": -0.2,
}

DANGEROUS_OBJECTS: dict[str, float] = {
    "gun": 0.40,
    "knife": 0.25,
    "baseball bat": 0.20,
    "scissors": 0.15,
    "bottle": 0.10,
}

# 区分テーブル: (係数上限, ラベル, HEXカラー)
CLASSIFY_TABLE = [
    (100, "執行対象外（クリア）", "#00BFFF"),
    (300, "潜在犯 / 要制圧", "#FF8C00"),
    (float("inf"), "執行対象 / 即時排除", "#FF0000"),
]

_yolo_detect_model = None
_yolo_pose_model = None


def _get_detect_model():
    global _yolo_detect_model
    if _yolo_detect_model is None:
        from ultralytics import YOLO
        _yolo_detect_model = YOLO("yolov8x.pt")
    return _yolo_detect_model


def _get_pose_model():
    global _yolo_pose_model
    if _yolo_pose_model is None:
        from ultralytics import YOLO
        _yolo_pose_model = YOLO("yolov8x-pose.pt")
    return _yolo_pose_model


def _run_detect(image_path: str) -> list[Box]:
    """YOLOv8x で全オブジェクト検出（confidence >= 0.5）"""
    model = _get_detect_model()
    results = model(image_path, verbose=False)
    boxes: list[Box] = []
    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            cls_name = result.names[cls_id]
            conf = float(box.conf[0])
            if conf >= 0.5:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                boxes.append({
                    "x1": int(x1),
                    "y1": int(y1),
                    "x2": int(x2),
                    "y2": int(y2),
                    "conf": conf,
                    "class": cls_name,
                })
    return boxes


def detect_persons(image_path: str) -> list[Box]:
    """YOLOv8x で人物のみ検出（confidence >= 0.5）"""
    return [b for b in _run_detect(image_path) if b["class"] == "person"]


def detect_all_objects(image_path: str) -> list[Box]:
    """YOLOv8x で全オブジェクト検出（context_score 用）"""
    return _run_detect(image_path)


def emotion_score(face_crop: np.ndarray) -> float:
    """DeepFace で感情スコア算出。clip(weighted_sum / 100, 0.0, 1.0)
    顔未検出: 0.15 / 感情取得失敗: 0.3
    """
    from deepface import DeepFace

    try:
        analysis = DeepFace.analyze(
            face_crop,
            actions=["emotion"],
            enforce_detection=False,
            silent=True,
        )
        if isinstance(analysis, list):
            analysis = analysis[0]
        emotions: dict[str, float] = analysis.get("emotion", {})
        if not emotions:
            return 0.3
        weighted_sum = sum(
            emotions.get(e, 0.0) * w for e, w in EMOTION_WEIGHTS.items()
        )
        return float(np.clip(weighted_sum / 100.0, 0.0, 1.0))
    except (ValueError, Exception) as exc:
        err_msg = str(exc).lower()
        if "face could not be detected" in err_msg or "no face" in err_msg:
            return 0.15
        return 0.3


def _angle_between(p_a: np.ndarray, p_b: np.ndarray, p_c: np.ndarray) -> float:
    """b を頂点とした角度 (度) を返す。a-b-c の角度。"""
    ba = p_a - p_b
    bc = p_c - p_b
    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)
    if norm_ba == 0 or norm_bc == 0:
        return 180.0
    cos_angle = np.dot(ba, bc) / (norm_ba * norm_bc)
    cos_angle = float(np.clip(cos_angle, -1.0, 1.0))
    return math.degrees(math.acos(cos_angle))


def posture_score(keypoints: np.ndarray, confidences: np.ndarray) -> float:
    """骨格テンション算出。confidence >= 0.5 のキーポイントのみ使用。
    有効キーポイント < 5: 0.2
    """
    valid_mask = confidences >= 0.5
    valid_count = int(valid_mask.sum())
    if valid_count < 5:
        return 0.2

    def get_kp(idx: int) -> np.ndarray | None:
        if idx < len(keypoints) and valid_mask[idx]:
            return keypoints[idx].copy()
        return None

    elbow_tensions: list[float] = []
    # 左: shoulder=5, elbow=7, wrist=9
    ls = get_kp(5)
    le = get_kp(7)
    lw = get_kp(9)
    if ls is not None and le is not None and lw is not None:
        deg = _angle_between(ls, le, lw)
        elbow_tensions.append((180.0 - deg) / 180.0)
    # 右: shoulder=6, elbow=8, wrist=10
    rs = get_kp(6)
    re = get_kp(8)
    rw = get_kp(10)
    if rs is not None and re is not None and rw is not None:
        deg = _angle_between(rs, re, rw)
        elbow_tensions.append((180.0 - deg) / 180.0)

    elbow_tension = float(np.mean(elbow_tensions)) if elbow_tensions else 0.0

    shoulder_asym = 0.0
    if ls is not None and rs is not None:
        kp_valid = keypoints[valid_mask]
        crop_height = float(kp_valid[:, 1].max() - kp_valid[:, 1].min())
        if crop_height > 0:
            shoulder_asym = min(abs(float(ls[1]) - float(rs[1])) / crop_height, 0.3)

    head_lean = 0.0
    nose = get_kp(0)
    if nose is not None and ls is not None and rs is not None:
        shoulder_mid_x = (float(ls[0]) + float(rs[0])) / 2.0
        shoulder_width = abs(float(rs[0]) - float(ls[0]))
        threshold_x = shoulder_width * 0.10
        if abs(float(nose[0]) - shoulder_mid_x) >= threshold_x:
            head_lean = 0.15

    score = elbow_tension * 0.6 + shoulder_asym * 0.4 + head_lean
    return float(np.clip(score, 0.0, 1.0))


def _boxes_overlap(box_a: Box, box_b: Box) -> bool:
    """2 つの Box が重なっているか判定する。"""
    return not (
        box_a["x2"] < box_b["x1"]
        or box_a["x1"] > box_b["x2"]
        or box_a["y2"] < box_b["y1"]
        or box_a["y1"] > box_b["y2"]
    )


def context_score(person_box: Box, all_boxes: list[Box], image_shape: tuple[int, int]) -> float:
    """人物クロップ 1.5 倍領域内の危険物を confidence >= 0.5 で検出・加点。
    clip(合計, 0.0, 1.0)
    """
    img_h, img_w = image_shape[:2]

    cx = (person_box["x1"] + person_box["x2"]) / 2.0
    cy = (person_box["y1"] + person_box["y2"]) / 2.0
    w = (person_box["x2"] - person_box["x1"]) * 1.5
    h = (person_box["y2"] - person_box["y1"]) * 1.5

    expanded: Box = {
        "x1": max(0, int(cx - w / 2)),
        "y1": max(0, int(cy - h / 2)),
        "x2": min(img_w, int(cx + w / 2)),
        "y2": min(img_h, int(cy + h / 2)),
        "conf": 1.0,
        "class": "_expanded",
    }

    total = 0.0
    for obj in all_boxes:
        if obj["conf"] < 0.5:
            continue
        cls = obj["class"]
        if cls not in DANGEROUS_OBJECTS:
            continue
        if not _boxes_overlap(expanded, obj):
            continue

        if cls == "bottle":
            # 信頼度 >= 0.7 かつ人物と重なる場合のみ加点
            if obj["conf"] >= 0.7 and _boxes_overlap(person_box, obj):
                total += DANGEROUS_OBJECTS["bottle"]
        else:
            total += DANGEROUS_OBJECTS[cls]

    return float(np.clip(total, 0.0, 1.0))


def classify(coeff: int, raw_score: float = 0.0) -> tuple[str, str]:
    """区分名と HEX カラーコードを返す。raw_score > 1.0 の場合は 測定不能。"""
    if raw_score > 1.0:
        return ("測定不能", "#8B008B")
    for threshold, label, color in CLASSIFY_TABLE:
        if coeff < threshold:
            return (label, color)
    return ("執行対象 / 即時排除", "#FF0000")



def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return (r, g, b)


def _draw_results(image_path: str, persons_data: list[dict]) -> str:
    """PIL で画像に bbox、係数、ラベルを描画して保存する。"""
    pil_img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(pil_img)

    font_large = None
    font_small = None
    font_candidates = [
        "/System/Library/Fonts/ヒラギノ角ゴシック W6.ttc",
        "/System/Library/Fonts/Hiragino Sans GB.ttc",
        "/System/Library/Fonts/Arial Unicode.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ]
    for font_path in font_candidates:
        if os.path.exists(font_path):
            try:
                font_large = ImageFont.truetype(font_path, 28)
                font_small = ImageFont.truetype(font_path, 20)
                break
            except Exception:
                continue
    if font_large is None:
        font_large = ImageFont.load_default()
        font_small = ImageFont.load_default()

    for person in persons_data:
        box = person["box"]
        coeff = person["coefficient"]
        label = person["label"]
        color_hex = person.get("color", "#FFFFFF")
        color_rgb = _hex_to_rgb(color_hex)

        x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]

        draw.rectangle([x1, y1, x2, y2], outline=color_rgb, width=3)

        text_coeff = f"係数: {coeff}"
        text_label = label
        bbox_coeff = draw.textbbox((0, 0), text_coeff, font=font_large)
        bbox_label = draw.textbbox((0, 0), text_label, font=font_small)
        text_w = max(bbox_coeff[2] - bbox_coeff[0], bbox_label[2] - bbox_label[0]) + 8
        text_h = (bbox_coeff[3] - bbox_coeff[1]) + (bbox_label[3] - bbox_label[1]) + 12

        bg_y1 = max(0, y1 - text_h)
        bg_y2 = y1
        bg_x2 = min(pil_img.width, x1 + text_w)

        draw.rectangle([x1, bg_y1, bg_x2, bg_y2], fill=color_rgb)
        draw.text((x1 + 4, bg_y1 + 2), text_coeff, fill=(255, 255, 255), font=font_large)
        draw.text((x1 + 4, bg_y1 + 2 + (bbox_coeff[3] - bbox_coeff[1]) + 2), text_label, fill=(255, 255, 255), font=font_small)

    src_path = Path(image_path)
    out_path = src_path.parent / f"{src_path.stem}_analyzed.jpg"
    pil_img.save(str(out_path), "JPEG", quality=95)
    return str(out_path)


def analyze(image_path: str) -> list[dict]:
    """フルパイプライン実行・画像保存。
    戻り値: [{"id": int, "coefficient": int, "label": str, "box": Box}]
    """
    detect_model = _get_detect_model()
    pose_model = _get_pose_model()

    bgr_img = cv2.imread(image_path)
    if bgr_img is None:
        raise FileNotFoundError(f"画像が読み込めません: {image_path}")
    img_shape = bgr_img.shape  # (H, W, C)

    person_boxes = detect_persons(image_path)
    if not person_boxes:
        print(f"[analyze] 人物が検出されませんでした: {image_path}")
        return []

    all_boxes = detect_all_objects(image_path)
    pose_results = pose_model(image_path, verbose=False)

    results: list[dict] = []

    for person_idx, pbox in enumerate(person_boxes):
        x1, y1, x2, y2 = pbox["x1"], pbox["y1"], pbox["x2"], pbox["y2"]

        face_h = (y2 - y1) // 3
        face_crop_bgr = bgr_img[y1:y1 + face_h, x1:x2]
        if face_crop_bgr.size == 0:
            face_crop_bgr = bgr_img[y1:y2, x1:x2]

        e_score = emotion_score(face_crop_bgr)

        p_score = 0.2
        if pose_results:
            best_pose_idx = None
            best_iou = -1.0
            for pr in pose_results:
                for bi, pose_box in enumerate(pr.boxes):
                    px1, py1, px2, py2 = pose_box.xyxy[0].tolist()
                    ix1 = max(x1, int(px1))
                    iy1 = max(y1, int(py1))
                    ix2 = min(x2, int(px2))
                    iy2 = min(y2, int(py2))
                    inter_w = max(0, ix2 - ix1)
                    inter_h = max(0, iy2 - iy1)
                    inter_area = inter_w * inter_h
                    union_area = (x2 - x1) * (y2 - y1) + (int(px2) - int(px1)) * (int(py2) - int(py1)) - inter_area
                    iou = inter_area / union_area if union_area > 0 else 0.0
                    if iou > best_iou:
                        best_iou = iou
                        best_pose_idx = (pr, bi)

            if best_pose_idx is not None and best_iou > 0.1:
                pr, bi = best_pose_idx
                if pr.keypoints is not None and bi < len(pr.keypoints):
                    kp_data = pr.keypoints[bi]
                    kp_xy = kp_data.xy[0].cpu().numpy()   # shape: [17, 2]
                    kp_conf = kp_data.conf[0].cpu().numpy()  # shape: [17]
                    p_score = posture_score(kp_xy, kp_conf)

        c_score = context_score(pbox, all_boxes, img_shape)

        raw_score = e_score * 0.40 + p_score * 0.35 + c_score * 0.25
        coeff = int(raw_score * 400)

        label, color = classify(coeff, raw_score)

        result_item = {
            "id": person_idx,
            "coefficient": coeff,
            "label": label,
            "color": color,
            "raw_score": raw_score,
            "emotion_score": e_score,
            "posture_score": p_score,
            "context_score": c_score,
            "box": pbox,
        }
        results.append(result_item)

        print(f"  [Person {person_idx}] 係数={coeff} ({label})")
        print(f"    emotion={e_score:.3f}  posture={p_score:.3f}  context={c_score:.3f}  raw={raw_score:.3f}")

    out_path = _draw_results(image_path, results)
    print(f"[analyze] 解析済み画像を保存: {out_path}")

    return results
