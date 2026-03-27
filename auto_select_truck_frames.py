import argparse
import csv
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO


BASE_DIR = Path(__file__).resolve().parent
TRUCK_MODEL_PATH = BASE_DIR / "Yolo-wight" / "truck.pt"
SIZE_MODEL_PATH = BASE_DIR / "Yolo-wight" / "best_size_March_25.pt"
FILL_SCRIPT_PATH = BASE_DIR / "size_estimation_v4.py"
CONF_THRESHOLD = 0.4
IOU_MATCH_THRESHOLD = 0.2
MAX_MISSED_SAMPLES = 2
MIN_TRACK_HITS = 2


@dataclass
class Detection:
    bbox: tuple[int, int, int, int]
    confidence: float
    score: float
    frame_index: int
    timestamp_sec: float
    image_path: Path


@dataclass
class Track:
    track_id: int
    bbox: tuple[int, int, int, int]
    hits: int
    missed: int
    best_detection: Detection
    selected: bool = False
    history: list[Detection] = field(default_factory=list)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Automatically select one best frame per truck and run fill estimation."
    )
    parser.add_argument("video_path", help="Path to the input video")
    parser.add_argument(
        "--fps",
        type=float,
        default=5.0,
        help="Sampling rate for candidate frames (default: 5)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for extracted best frames and results",
    )
    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="Disable the live detection preview window",
    )
    return parser.parse_args()


def compute_iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter_area

    if union <= 0:
        return 0.0

    return inter_area / union


def blur_score(frame, bbox):
    x1, y1, x2, y2 = bbox
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return 0.0

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return min(variance / 400.0, 1.0)


def edge_penalty(bbox, frame_width, frame_height, margin_ratio=0.03):
    x1, y1, x2, y2 = bbox
    x_margin = frame_width * margin_ratio
    y_margin = frame_height * margin_ratio

    touches_edge = (
        x1 <= x_margin
        or y1 <= y_margin
        or x2 >= frame_width - x_margin
        or y2 >= frame_height - y_margin
    )
    return 1.0 if touches_edge else 0.0


def detection_score(frame, bbox, confidence):
    frame_height, frame_width = frame.shape[:2]
    x1, y1, x2, y2 = bbox

    box_area = max(0, x2 - x1) * max(0, y2 - y1)
    frame_area = frame_width * frame_height
    area_score = box_area / frame_area if frame_area else 0.0

    box_center_x = (x1 + x2) / 2.0
    box_center_y = (y1 + y2) / 2.0
    frame_center_x = frame_width / 2.0
    frame_center_y = frame_height / 2.0
    distance = ((box_center_x - frame_center_x) ** 2 + (box_center_y - frame_center_y) ** 2) ** 0.5
    max_distance = (frame_center_x ** 2 + frame_center_y ** 2) ** 0.5
    center_score = 1.0 - (distance / max_distance if max_distance else 0.0)

    sharpness_score = blur_score(frame, bbox)
    border_penalty = edge_penalty(bbox, frame_width, frame_height)

    return (
        confidence
        + (2.0 * area_score)
        + center_score
        + sharpness_score
        - (1.5 * border_penalty)
    )


def finalize_track(track, completed_tracks):
    if track.hits >= MIN_TRACK_HITS and track.best_detection is not None:
        track.selected = True
        completed_tracks.append(track)


def resize_mask(mask, target_shape):
    return cv2.resize(
        mask.astype(np.uint8),
        (target_shape[1], target_shape[0]),
        interpolation=cv2.INTER_NEAREST,
    ).astype(bool)


def apply_segmentation_overlay(frame, bbox, size_model, size_classes, device):
    x1, y1, x2, y2 = bbox
    truck_crop = frame[y1:y2, x1:x2]
    if truck_crop.size == 0:
        return

    seg_result = size_model(truck_crop, device=device, verbose=False)[0]
    if seg_result.masks is None:
        return

    truck_box_mask = None
    content_mask = None
    masks = seg_result.masks.data.cpu().numpy()
    classes = seg_result.boxes.cls.cpu().numpy()

    for index, cls in enumerate(classes):
        class_name = size_classes[int(cls)]
        if class_name.lower() == "box":
            truck_box_mask = masks[index]
        elif class_name.lower() == "content":
            content_mask = masks[index]

    if truck_box_mask is None:
        return

    overlay = truck_crop.copy()
    box_mask_resized = resize_mask(truck_box_mask, truck_crop.shape)
    overlay[box_mask_resized] = (255, 0, 0)

    if content_mask is not None:
        content_mask_resized = resize_mask(content_mask, truck_crop.shape)
        overlay[content_mask_resized] = (0, 255, 0)

    truck_crop[:] = cv2.addWeighted(overlay, 0.4, truck_crop, 0.6, 0)


def detect_trucks(model, frame, truck_classes, device):
    detections = []
    results = model(frame, device=device, verbose=False)[0]

    for det in results.boxes:
        confidence = float(det.conf[0])
        if confidence < CONF_THRESHOLD:
            continue

        class_id = int(det.cls[0])
        if truck_classes[class_id] != "truck":
            continue

        bbox = tuple(map(int, det.xyxy[0]))
        score = detection_score(frame, bbox, confidence)
        detections.append((bbox, confidence, score))

    detections.sort(key=lambda item: item[2], reverse=True)
    return detections


def save_frame(output_dir, track_id, frame_index, frame):
    image_path = output_dir / f"truck_{track_id:03d}_frame_{frame_index:05d}.jpg"
    cv2.imwrite(str(image_path), frame)
    return image_path


def resize_for_display(image, max_width=1280, max_height=720):
    height, width = image.shape[:2]
    scale = min(max_width / width, max_height / height, 1.0)
    if scale == 1.0:
        return image

    new_size = (int(width * scale), int(height * scale))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)


def draw_preview(frame, active_tracks, sampled_frame_index, size_model, size_classes, device):
    preview = frame.copy()
    for track in active_tracks.values():
        x1, y1, x2, y2 = track.bbox
        apply_segmentation_overlay(preview, track.bbox, size_model, size_classes, device)
        cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 255, 0), 3)
        label = f"truck_{track.track_id:03d} best={track.best_detection.score:.2f}"
        cv2.putText(
            preview,
            label,
            (x1, max(30, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2,
        )

    cv2.putText(
        preview,
        f"sampled frame: {sampled_frame_index}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 255),
        2,
    )
    return resize_for_display(preview)


def match_tracks(active_tracks, detections):
    matches = []
    unmatched_track_ids = set(active_tracks.keys())
    unmatched_detection_ids = set(range(len(detections)))

    scored_pairs = []
    for track_id, track in active_tracks.items():
        for detection_index, (bbox, _, _) in enumerate(detections):
            iou = compute_iou(track.bbox, bbox)
            if iou >= IOU_MATCH_THRESHOLD:
                scored_pairs.append((iou, track_id, detection_index))

    for _, track_id, detection_index in sorted(scored_pairs, reverse=True):
        if track_id not in unmatched_track_ids or detection_index not in unmatched_detection_ids:
            continue
        matches.append((track_id, detection_index))
        unmatched_track_ids.remove(track_id)
        unmatched_detection_ids.remove(detection_index)

    return matches, unmatched_track_ids, unmatched_detection_ids


def run_fill_script(image_path):
    env = os.environ.copy()
    env["YOLO_NO_DISPLAY"] = "1"
    command = [sys.executable, str(FILL_SCRIPT_PATH), str(image_path)]
    completed = subprocess.run(
        command,
        cwd=str(BASE_DIR),
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )

    output = completed.stdout + completed.stderr
    match = re.search(r"FINAL FILL:\s*([0-9]+(?:\.[0-9]+)?)%", output)
    fill_percentage = float(match.group(1)) if match else None
    return completed.returncode, fill_percentage, output


def analyze_video(video_path, output_dir, sampling_fps, show_preview):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    truck_model = YOLO(str(TRUCK_MODEL_PATH))
    size_model = YOLO(str(SIZE_MODEL_PATH))
    truck_classes = truck_model.names
    size_classes = size_model.names

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    source_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_interval = max(int(round(source_fps / sampling_fps)), 1)

    active_tracks = {}
    completed_tracks = []
    next_track_id = 1
    frame_index = 0

    while True:
        success, frame = cap.read()
        if not success:
            break

        if frame_index % frame_interval != 0:
            frame_index += 1
            continue

        timestamp_sec = frame_index / source_fps
        detections = detect_trucks(truck_model, frame, truck_classes, device)
        matches, unmatched_track_ids, unmatched_detection_ids = match_tracks(active_tracks, detections)

        for track_id, detection_index in matches:
            bbox, confidence, score = detections[detection_index]
            track = active_tracks[track_id]
            image_path = save_frame(output_dir, track_id, frame_index, frame)
            detection = Detection(bbox, confidence, score, frame_index, timestamp_sec, image_path)
            track.bbox = bbox
            track.hits += 1
            track.missed = 0
            track.history.append(detection)
            if score > track.best_detection.score:
                track.best_detection = detection

        for track_id in list(unmatched_track_ids):
            track = active_tracks[track_id]
            track.missed += 1
            if track.missed > MAX_MISSED_SAMPLES:
                finalize_track(track, completed_tracks)
                del active_tracks[track_id]

        for detection_index in unmatched_detection_ids:
            bbox, confidence, score = detections[detection_index]
            image_path = save_frame(output_dir, next_track_id, frame_index, frame)
            detection = Detection(bbox, confidence, score, frame_index, timestamp_sec, image_path)
            active_tracks[next_track_id] = Track(
                track_id=next_track_id,
                bbox=bbox,
                hits=1,
                missed=0,
                best_detection=detection,
                history=[detection],
            )
            next_track_id += 1

        if show_preview:
            preview = draw_preview(frame, active_tracks, frame_index, size_model, size_classes, device)
            cv2.imshow("Truck Detection Preview", preview)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

        frame_index += 1

    cap.release()
    if show_preview:
        cv2.destroyAllWindows()

    for track in list(active_tracks.values()):
        finalize_track(track, completed_tracks)

    return completed_tracks


def write_summary(output_dir, rows):
    summary_path = output_dir / "truck_fill_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "truck_id",
                "frame_index",
                "timestamp_sec",
                "selected_frame",
                "fill_percentage",
                "status",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    return summary_path


def write_all_frames_fill_csv(output_dir, completed_tracks):
    csv_path = output_dir / "all_frame_fill_levels.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "truck_id",
                "frame_index",
                "timestamp_sec",
                "frame_path",
                "is_selected_frame",
                "fill_percentage",
                "status",
            ],
        )
        writer.writeheader()

        for track in sorted(completed_tracks, key=lambda item: item.track_id):
            best_frame_path = str(track.best_detection.image_path)
            for detection in sorted(track.history, key=lambda item: item.frame_index):
                return_code, fill_percentage, _ = run_fill_script(detection.image_path)
                status = "ok" if return_code == 0 and fill_percentage is not None else "failed"
                writer.writerow(
                    {
                        "truck_id": track.track_id,
                        "frame_index": detection.frame_index,
                        "timestamp_sec": f"{detection.timestamp_sec:.2f}",
                        "frame_path": str(detection.image_path),
                        "is_selected_frame": "yes" if str(detection.image_path) == best_frame_path else "no",
                        "fill_percentage": "" if fill_percentage is None else f"{fill_percentage:.2f}",
                        "status": status,
                    }
                )

    return csv_path


def main():
    args = parse_args()
    video_path = Path(args.video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    output_dir = args.output_dir or (BASE_DIR / "auto_outputs" / video_path.stem)
    output_dir.mkdir(parents=True, exist_ok=True)

    completed_tracks = analyze_video(video_path, output_dir, args.fps, not args.no_preview)
    if not completed_tracks:
        print("No truck tracks found.")
        return

    summary_rows = []

    for track in sorted(completed_tracks, key=lambda item: item.track_id):
        best = track.best_detection
        return_code, fill_percentage, raw_output = run_fill_script(best.image_path)
        status = "ok" if return_code == 0 and fill_percentage is not None else "failed"
        summary_rows.append(
            {
                "truck_id": track.track_id,
                "frame_index": best.frame_index,
                "timestamp_sec": f"{best.timestamp_sec:.2f}",
                "selected_frame": str(best.image_path),
                "fill_percentage": "" if fill_percentage is None else f"{fill_percentage:.2f}",
                "status": status,
            }
        )

        print("##########################")
        print(f"Selected truck frame: {best.image_path}")
        print(f"Fill level: {fill_percentage:.2f}%" if fill_percentage is not None else "Fill level: N/A")

        log_path = output_dir / f"truck_{track.track_id:03d}_fill_output.txt"
        log_path.write_text(raw_output, encoding="utf-8")

    summary_path = write_summary(output_dir, summary_rows)
    all_frames_csv_path = write_all_frames_fill_csv(output_dir, completed_tracks)
    print("##########################")
    print(f"Summary written to: {summary_path}")
    print(f"All-frame CSV written to: {all_frames_csv_path}")


if __name__ == "__main__":
    main()
