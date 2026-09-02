#!/usr/bin/env python3
"""
Compare sharpness across images or videos and generate a visual report.

Typical use:
python3 cap-com/compare_image_sharpness.py \
  --input raw.mp4 \
  --input calibrated.mp4 \
  --input long_video.mp4 \
  --input edge_frame.jpg \
  --time-seconds 10 \
  --output-dir cap-com/_sharpness_compare
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

cv2 = None
np = None


def ensure_image_dependencies() -> None:
    global cv2, np
    if cv2 is not None and np is not None:
        return
    try:
        import cv2 as cv2_module
        import numpy as np_module
    except ModuleNotFoundError as exc:
        missing = exc.name or "opencv-python/numpy"
        raise SystemExit(
            f"Missing Python package: {missing}. "
            "Install dependencies with: python3 -m pip install -r requirements.txt"
        ) from exc
    cv2 = cv2_module
    np = np_module


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".avi", ".m4v", ".webm"}


@dataclass(frozen=True)
class LoadedFrame:
    source: Path
    label: str
    kind: str
    frame: np.ndarray
    frame_index: int | None
    time_seconds: float | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare sharpness across images/videos and generate side-by-side outputs.",
    )
    parser.add_argument(
        "--input",
        dest="inputs",
        action="append",
        type=Path,
        required=True,
        help="Image or video path. Use multiple times in pipeline order.",
    )
    parser.add_argument(
        "--label",
        dest="labels",
        action="append",
        default=[],
        help="Optional display label for each input, in the same order as --input.",
    )
    parser.add_argument(
        "--time-seconds",
        type=float,
        default=0.0,
        help="Frame time for video inputs. Default: 0",
    )
    parser.add_argument(
        "--frame-index",
        type=int,
        default=None,
        help="Exact frame index for video inputs. Overrides --time-seconds.",
    )
    parser.add_argument(
        "--last-frame",
        action="store_true",
        help="Use the last decodable frame from each video.",
    )
    parser.add_argument(
        "--crop",
        default=None,
        help="Optional ROI shared by all inputs: x,y,w,h",
    )
    parser.add_argument(
        "--zoom-size",
        type=int,
        default=256,
        help="Square crop size for the zoom-strip preview. Default: 256",
    )
    parser.add_argument(
        "--preview-height",
        type=int,
        default=360,
        help="Target height for the main comparison strip. Default: 360",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("cap-com/_sharpness_compare"),
        help="Output folder for report files.",
    )
    return parser.parse_args()


def parse_crop(crop_text: str | None) -> tuple[int, int, int, int] | None:
    if not crop_text:
        return None
    parts = [part.strip() for part in crop_text.split(",")]
    if len(parts) != 4:
        raise ValueError("--crop must be x,y,w,h")
    x, y, w, h = [int(part) for part in parts]
    if w <= 0 or h <= 0:
        raise ValueError("--crop width and height must be greater than 0")
    return x, y, w, h


def detect_kind(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in IMAGE_SUFFIXES:
        return "image"
    if suffix in VIDEO_SUFFIXES:
        return "video"
    raise ValueError(f"Unsupported file type: {path}")


def resolve_label(path: Path, labels: list[str], index: int) -> str:
    if index < len(labels) and labels[index].strip():
        return labels[index].strip()
    return path.name


def load_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Failed to read image: {path}")
    return image


def load_video_frame(path: Path, time_seconds: float, frame_index: int | None, last_frame: bool) -> tuple[np.ndarray, int | None, float | None]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if fps and fps > 0 else None
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    target_index: int | None = None
    if last_frame and total_frames > 0:
        target_index = max(0, total_frames - 1)
    elif frame_index is not None:
        target_index = max(0, frame_index)
    elif fps is not None:
        target_index = max(0, int(round(time_seconds * fps)))

    if target_index is not None:
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_index)
    elif time_seconds > 0:
        cap.set(cv2.CAP_PROP_POS_MSEC, time_seconds * 1000.0)

    ok, frame = cap.read()
    if not ok or frame is None:
        cap.release()
        raise RuntimeError(f"Failed to decode frame from video: {path}")

    actual_index = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
    if actual_index < 0:
        actual_index = target_index

    if fps is not None and actual_index is not None and actual_index >= 0:
        actual_time = actual_index / fps
    else:
        actual_time = time_seconds

    cap.release()
    return frame, actual_index, actual_time


def load_input(path: Path, label: str, time_seconds: float, frame_index: int | None, last_frame: bool) -> LoadedFrame:
    kind = detect_kind(path)
    if kind == "image":
        return LoadedFrame(
            source=path,
            label=label,
            kind=kind,
            frame=load_image(path),
            frame_index=None,
            time_seconds=None,
        )
    frame, actual_index, actual_time = load_video_frame(path, time_seconds, frame_index, last_frame)
    return LoadedFrame(
        source=path,
        label=label,
        kind=kind,
        frame=frame,
        frame_index=actual_index,
        time_seconds=actual_time,
    )


def apply_crop(image: np.ndarray, crop: tuple[int, int, int, int] | None) -> np.ndarray:
    if crop is None:
        return image
    x, y, w, h = crop
    image_h, image_w = image.shape[:2]
    x = max(0, min(x, image_w - 1))
    y = max(0, min(y, image_h - 1))
    w = min(w, image_w - x)
    h = min(h, image_h - y)
    return image[y:y + h, x:x + w]


def to_gray(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def variance_of_laplacian(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def tenengrad(gray: np.ndarray) -> float:
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    return float(np.mean(gx * gx + gy * gy))


def brenner(gray: np.ndarray) -> float:
    diff = gray[:, 2:].astype(np.float32) - gray[:, :-2].astype(np.float32)
    return float(np.mean(diff * diff))


def mean_gradient(gray: np.ndarray) -> float:
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    return float(np.mean(mag))


def estimate_noise(gray: np.ndarray) -> float:
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    residual = gray.astype(np.float32) - blur.astype(np.float32)
    return float(np.std(residual))


def compute_metrics(image: np.ndarray) -> dict[str, float]:
    gray = to_gray(image)
    return {
        "laplacian_var": variance_of_laplacian(gray),
        "tenengrad": tenengrad(gray),
        "brenner": brenner(gray),
        "mean_gradient": mean_gradient(gray),
        "noise_std": estimate_noise(gray),
    }


def resize_to_match(image: np.ndarray, width: int, height: int) -> np.ndarray:
    if image.shape[1] == width and image.shape[0] == height:
        return image
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_CUBIC)


def psnr(reference: np.ndarray, candidate: np.ndarray) -> float:
    ref = reference.astype(np.float32)
    cur = candidate.astype(np.float32)
    mse = float(np.mean((ref - cur) ** 2))
    if mse <= 1e-12:
        return float("inf")
    return float(20.0 * math.log10(255.0 / math.sqrt(mse)))


def simple_ssim(reference_gray: np.ndarray, candidate_gray: np.ndarray) -> float:
    ref = reference_gray.astype(np.float64)
    cur = candidate_gray.astype(np.float64)
    c1 = 6.5025
    c2 = 58.5225

    mu1 = cv2.GaussianBlur(ref, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(cur, (11, 11), 1.5)
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = cv2.GaussianBlur(ref * ref, (11, 11), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(cur * cur, (11, 11), 1.5) - mu2_sq
    sigma12 = cv2.GaussianBlur(ref * cur, (11, 11), 1.5) - mu1_mu2

    numerator = (2 * mu1_mu2 + c1) * (2 * sigma12 + c2)
    denominator = (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    score = numerator / np.maximum(denominator, 1e-12)
    return float(np.mean(score))


def center_crop_square(image: np.ndarray, size: int) -> np.ndarray:
    h, w = image.shape[:2]
    side = min(size, h, w)
    x = max(0, (w - side) // 2)
    y = max(0, (h - side) // 2)
    return image[y:y + side, x:x + side]


def annotate_tile(image: np.ndarray, lines: list[str]) -> np.ndarray:
    top_pad = 96
    out = cv2.copyMakeBorder(image, top_pad, 0, 0, 0, cv2.BORDER_CONSTANT, value=(18, 18, 18))
    y = 28
    for index, line in enumerate(lines):
        scale = 0.75 if index == 0 else 0.58
        thickness = 2 if index == 0 else 1
        cv2.putText(out, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (240, 240, 240), thickness, cv2.LINE_AA)
        y += 22
    return out


def build_main_strip(frames: list[LoadedFrame], metrics_list: list[dict[str, float]], preview_height: int) -> np.ndarray:
    tiles: list[np.ndarray] = []
    for frame, metrics in zip(frames, metrics_list):
        image = frame.frame
        scale = preview_height / image.shape[0]
        preview_width = max(1, int(round(image.shape[1] * scale)))
        resized = cv2.resize(image, (preview_width, preview_height), interpolation=cv2.INTER_AREA)
        meta = [frame.label, f"{image.shape[1]}x{image.shape[0]}", f"lap={metrics['laplacian_var']:.1f}", f"ten={metrics['tenengrad']:.1f}"]
        tiles.append(annotate_tile(resized, meta))
    return cv2.hconcat(tiles)


def build_zoom_strip(frames: list[LoadedFrame], metrics_list: list[dict[str, float]], zoom_size: int) -> np.ndarray:
    tiles: list[np.ndarray] = []
    for frame, metrics in zip(frames, metrics_list):
        crop = center_crop_square(frame.frame, zoom_size)
        if crop.shape[0] != zoom_size or crop.shape[1] != zoom_size:
            crop = resize_to_match(crop, zoom_size, zoom_size)
        meta = [f"{frame.label} zoom", f"grad={metrics['mean_gradient']:.2f}", f"noise={metrics['noise_std']:.2f}"]
        tiles.append(annotate_tile(crop, meta))
    return cv2.hconcat(tiles)


def main() -> int:
    args = parse_args()
    ensure_image_dependencies()
    crop = parse_crop(args.crop)

    if args.labels and len(args.labels) != len(args.inputs):
        raise SystemExit("If --label is used, provide one label per --input.")

    loaded_frames: list[LoadedFrame] = []
    for index, input_path in enumerate(args.inputs):
        path = input_path.resolve()
        if not path.exists():
            raise SystemExit(f"Input not found: {path}")
        label = resolve_label(path, args.labels, index)
        loaded = load_input(path, label, args.time_seconds, args.frame_index, args.last_frame)
        loaded_frames.append(
            LoadedFrame(
                source=loaded.source,
                label=loaded.label,
                kind=loaded.kind,
                frame=apply_crop(loaded.frame, crop),
                frame_index=loaded.frame_index,
                time_seconds=loaded.time_seconds,
            )
        )

    if len(loaded_frames) < 2:
        raise SystemExit("Provide at least two inputs to compare.")

    metrics_list = [compute_metrics(item.frame) for item in loaded_frames]

    reference = loaded_frames[0].frame
    ref_h, ref_w = reference.shape[:2]
    reference_gray = to_gray(reference)
    comparison_rows = []
    for item, metrics in zip(loaded_frames, metrics_list):
        resized = resize_to_match(item.frame, ref_w, ref_h)
        row = {
            "label": item.label,
            "path": str(item.source),
            "kind": item.kind,
            "frame_index": item.frame_index,
            "time_seconds": item.time_seconds,
            "width": int(item.frame.shape[1]),
            "height": int(item.frame.shape[0]),
            "metrics": metrics,
            "vs_reference": {
                "psnr": psnr(reference, resized),
                "ssim": simple_ssim(reference_gray, to_gray(resized)),
                "laplacian_ratio": metrics["laplacian_var"] / max(metrics_list[0]["laplacian_var"], 1e-12),
                "tenengrad_ratio": metrics["tenengrad"] / max(metrics_list[0]["tenengrad"], 1e-12),
            },
        }
        comparison_rows.append(row)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    main_strip = build_main_strip(loaded_frames, metrics_list, args.preview_height)
    zoom_strip = build_zoom_strip(loaded_frames, metrics_list, args.zoom_size)

    strip_path = output_dir / "comparison_strip.jpg"
    zoom_path = output_dir / "comparison_zoom_strip.jpg"
    report_path = output_dir / "report.json"

    cv2.imwrite(str(strip_path), main_strip, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    cv2.imwrite(str(zoom_path), zoom_strip, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    report = {
        "reference": comparison_rows[0]["label"],
        "crop": crop,
        "time_seconds": args.time_seconds,
        "frame_index": args.frame_index,
        "last_frame": args.last_frame,
        "results": comparison_rows,
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"Saved: {strip_path}")
    print(f"Saved: {zoom_path}")
    print(f"Saved: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
