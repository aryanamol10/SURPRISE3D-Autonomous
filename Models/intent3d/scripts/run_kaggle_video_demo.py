"""Run the Kaggle dash-cam dataset through the raw-video branch."""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
MODELS_DIR = SCRIPT_DIR.parent / "models"
sys.path.insert(0, str(MODELS_DIR))

try:
    from torchvision.models.detection import (
        SSDLite320_MobileNet_V3_Large_Weights,
        ssdlite320_mobilenet_v3_large,
    )
except Exception as exc:
    raise RuntimeError("torchvision detection models are required for this demo") from exc

try:
    from PIL import Image
except Exception:
    Image = None

import cv2


def _try_import_kagglehub():
    try:
        import kagglehub
    except Exception as exc:
        raise RuntimeError("Install kagglehub first: pip install kagglehub") from exc
    return kagglehub


def _collect_media_files(root_dir):
    root_dir = Path(root_dir)
    if root_dir.is_file():
        return [root_dir.resolve()]

    files = []
    for extension in (
        "*.mp4", "*.avi", "*.mov", "*.mkv", "*.webm",
        "*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp",
    ):
        files.extend(root_dir.rglob(extension))
    return sorted({path.resolve() for path in files})


def _media_to_frame_tensor(media_path):
    suffix = Path(media_path).suffix.lower()
    if suffix in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
        if Image is None:
            raise RuntimeError("PIL is required to decode image-based datasets")
        image = Image.open(media_path).convert("RGB")
        return torch.from_numpy(np.asarray(image))

    frames, _ = _decode_video(media_path)
    return frames


def _download_dataset(dataset_slug):
    kagglehub = _try_import_kagglehub()
    return Path(kagglehub.dataset_download(dataset_slug)).resolve()


def _decode_video(video_path):
    suffix = Path(video_path).suffix.lower()
    if suffix in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
        if Image is None:
            raise RuntimeError("PIL is required to decode image-based datasets")
        image = Image.open(video_path).convert("RGB")
        frame = np.asarray(image)
        timestamp = np.asarray([0.0], dtype=np.float32)
        return torch.from_numpy(frame[None, ...]), timestamp

    try:
        from torchvision.io import read_video

        frames, _, info = read_video(str(video_path), pts_unit="sec")
        if frames.numel() > 0:
            fps = float(info.get("video_fps", 0.0) or 0.0)
            timestamps = np.arange(len(frames), dtype=np.float32)
            if fps > 0:
                timestamps = timestamps / fps
            return frames, timestamps
    except Exception:
        pass

    try:
        import cv2

        capture = cv2.VideoCapture(str(video_path))
        frames = []
        timestamps = []
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_index = 0
        while True:
            success, frame = capture.read()
            if not success:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
            timestamps.append(frame_index / fps if fps > 0 else float(frame_index))
            frame_index += 1
        capture.release()
        if frames:
            return torch.from_numpy(np.stack(frames, axis=0)), np.asarray(timestamps, dtype=np.float32)
    except Exception:
        pass

    raise RuntimeError(f"Could not decode media file: {video_path}")


def _iter_frames(media_path):
    suffix = Path(media_path).suffix.lower()
    if suffix in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
        frames, timestamps = _decode_video(media_path)
        yield frames[0], 0, timestamps[0] if len(timestamps) else 0.0
        return

    capture = cv2.VideoCapture(str(media_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open media file: {media_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_index = 0
    try:
        while True:
            success, frame = capture.read()
            if not success:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            timestamp = frame_index / fps if fps > 0 else float(frame_index)
            yield torch.from_numpy(frame.copy()), frame_index, timestamp
            frame_index += 1
    finally:
        capture.release()


def _load_detector(device):
    weights = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
    detector = ssdlite320_mobilenet_v3_large(weights=weights)
    detector.to(device)
    detector.eval()
    return detector, weights


def _prepare_detector_input(frame_tensor):
    if frame_tensor.dtype != torch.float32:
        frame_tensor = frame_tensor.float()
    if frame_tensor.max() > 1.5:
        frame_tensor = frame_tensor / 255.0
    return frame_tensor.permute(2, 0, 1).contiguous()


def _distance_bin_from_box(box, image_width, image_height, num_bins):
    x1, y1, x2, y2 = box
    box_w = max(float(x2 - x1), 1.0)
    box_h = max(float(y2 - y1), 1.0)
    area_ratio = (box_w * box_h) / max(float(image_width * image_height), 1.0)
    area_score = 1.0 - min(max(area_ratio * 8.0, 0.0), 1.0)
    bottom_score = 1.0 - min(max(float((y1 + y2) * 0.5) / max(image_height, 1), 0.0), 1.0)
    combined = 0.7 * area_score + 0.3 * bottom_score
    class_idx = int(np.clip(np.floor(combined * num_bins), 0, num_bins - 1))
    return class_idx, combined


def _color_for_class(class_idx):
    palette = [
        (46, 204, 113),
        (52, 152, 219),
        (241, 196, 15),
        (230, 126, 34),
        (231, 76, 60),
        (155, 89, 182),
        (26, 188, 156),
        (149, 165, 166),
    ]
    return palette[class_idx % len(palette)]


def _draw_detections(frame_bgr, detections, distance_classes):
    overlay = frame_bgr.copy()
    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det["box"]]
        color = _color_for_class(det["distance_class"])
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
        label = f'{det["label"]}: {det["distance_label"]} ({det["score"]:.2f})'
        cv2.putText(
            overlay,
            label,
            (x1, max(y1 - 6, 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )
    return overlay


def _write_mp4(output_path, frames_bgr, fps=8.0):
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not frames_bgr:
        raise RuntimeError("No frames were rendered, so no MP4 can be written")

    height, width = frames_bgr[0].shape[:2]
    codecs = ["mp4v", "avc1", "H264", "XVID"]
    writer = None
    chosen_codec = None
    for codec in codecs:
        candidate = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*codec),
            fps,
            (width, height),
        )
        if candidate.isOpened():
            writer = candidate
            chosen_codec = codec
            break

    if writer is None:
        raise RuntimeError(
            f"Could not open a video writer for {output_path}. Tried codecs: {', '.join(codecs)}"
        )

    for frame_bgr in frames_bgr:
        writer.write(np.ascontiguousarray(frame_bgr))
    writer.release()
    return output_path, chosen_codec


def _sample_frames(frames, timestamps, num_points, frame_size):
    if frames.dim() == 4:
        total_frames = frames.shape[0]
        if total_frames <= num_points:
            sampled_inds = np.arange(total_frames, dtype=np.int64)
        else:
            sampled_inds = np.linspace(0, total_frames - 1, num_points).astype(np.int64)

        frames = frames[sampled_inds]
        if frames.dtype != torch.float32:
            frames = frames.float()
        if frames.max() > 1.5:
            frames = frames / 255.0
        frames = frames.permute(0, 3, 1, 2).contiguous()
        frames = torch.nn.functional.interpolate(
            frames,
            size=(frame_size, frame_size),
            mode="bilinear",
            align_corners=False,
        )
        frame_mask = torch.ones((frames.shape[0],), dtype=torch.bool)
        sampled_timestamps = torch.from_numpy(np.asarray(timestamps)[sampled_inds]).float()
        return frames, frame_mask, sampled_timestamps

    raise RuntimeError("Decoded frames must have shape (T, H, W, C)")


class VideoDemoModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, distance_classes):
        super().__init__()
        self.encoder = VideoEncoder(
            input_dim=input_dim,
            d_model=hidden_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
        )
        self.head = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(hidden_dim, distance_classes),
        )

    def forward(self, video_frames=None, video_features=None, video_mask=None, video_timestamps=None):
        frame_features, pooled = self.encoder(
            video_frames=video_frames,
            video_features=video_features,
            video_mask=video_mask,
            video_timestamps=video_timestamps,
        )
        logits = self.head(frame_features)
        return {
            "video_feats": frame_features,
            "video_pooled": pooled,
            "video_distance_logits": logits,
        }


def _load_model(args, device):
    model = VideoDemoModel(
        input_dim=args.video_input_dim,
        hidden_dim=args.video_hidden_dim,
        num_layers=args.video_num_layers,
        distance_classes=args.video_distance_classes,
    )
    model.to(device)
    model.eval()

    if args.checkpoint_path:
        checkpoint = torch.load(args.checkpoint_path, map_location=device)
        state_dict = checkpoint.get("model", checkpoint)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"Loaded checkpoint with missing keys: {len(missing)}, unexpected keys: {len(unexpected)}")

    return model


def _run_detection_pipeline(args, media_files, device):
    detector, weights = _load_detector(device)
    category_names = weights.meta.get("categories", [])
    transform = weights.transforms()

    predictions = []
    rendered_frames = []

    for media_path in media_files[: args.max_videos]:
        for frame, frame_index, timestamp in _iter_frames(media_path):
            frame_uint8 = frame.numpy().astype(np.uint8)
            frame_bgr = cv2.cvtColor(frame_uint8, cv2.COLOR_RGB2BGR)

            detector_input = transform(_prepare_detector_input(frame))
            with torch.no_grad():
                outputs = detector([detector_input.to(device)])[0]

            boxes = outputs["boxes"].detach().cpu().numpy()
            labels = outputs["labels"].detach().cpu().numpy()
            scores = outputs["scores"].detach().cpu().numpy()

            detections = []
            for box, label, score in zip(boxes, labels, scores):
                if score < args.detection_threshold:
                    continue
                distance_class, distance_score = _distance_bin_from_box(
                    box, frame_bgr.shape[1], frame_bgr.shape[0], args.video_distance_classes
                )
                detections.append(
                    {
                        "box": box.tolist(),
                        "label": category_names[label] if label < len(category_names) else str(int(label)),
                        "score": float(score),
                        "distance_class": int(distance_class),
                        "distance_label": f"dist_{distance_class}",
                        "distance_score": float(distance_score),
                    }
                )

            predictions.append(
                {
                    "media_path": str(media_path),
                    "frame_index": int(frame_index),
                    "timestamp": float(timestamp),
                    "detections": detections,
                }
            )

            if args.output_mp4:
                rendered = _draw_detections(frame_bgr, detections, args.video_distance_classes)
                rendered_frames.append(rendered)

    return predictions, rendered_frames


def main():
    parser = argparse.ArgumentParser(description="Run the Kaggle video dataset through the raw-video branch.")
    parser.add_argument(
        "--dataset",
        default="maadaaai/sunny-day-city-road-dash-cam-video-dataset",
        help="Kaggle dataset slug.",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download the Kaggle dataset before running.",
    )
    parser.add_argument(
        "--video_root",
        default=None,
        help="Root directory, image folder, or single media file to process.",
    )
    parser.add_argument(
        "--checkpoint_path",
        default=None,
        help="Optional checkpoint to load video branch weights from.",
    )
    parser.add_argument(
        "--output_json",
        default="video_predictions.json",
        help="Where to write the predictions.",
    )
    parser.add_argument(
        "--output_mp4",
        default="detections.mp4",
        help="Where to write the annotated video.",
    )
    parser.add_argument("--max_videos", type=int, default=20)
    parser.add_argument("--video_distance_classes", type=int, default=8)
    parser.add_argument("--detection_threshold", type=float, default=0.5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if args.download:
        dataset_root = _download_dataset(args.dataset)
    elif args.video_root:
        dataset_root = Path(args.video_root).resolve()
    else:
        raise ValueError("Provide either --download or --video_root")

    video_files = _collect_media_files(dataset_root)
    if not video_files:
        raise RuntimeError(f"No video or image files found under {dataset_root}")

    device = torch.device(args.device)

    predictions, rendered_frames = _run_detection_pipeline(args, video_files, device)

    for item in predictions[: min(len(predictions), 10)]:
        detection_count = len(item["detections"])
        print(f"{Path(item['media_path']).name}: {detection_count} detections")

    output_path = Path(args.output_json).resolve()
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(predictions, handle, indent=2)

    print(f"Saved predictions to {output_path}")

    if rendered_frames and args.output_mp4:
        output_video_path, chosen_codec = _write_mp4(args.output_mp4, rendered_frames, fps=8.0)
        print(f"Saved annotated video to {output_video_path} using codec {chosen_codec}")


if __name__ == "__main__":
    main()