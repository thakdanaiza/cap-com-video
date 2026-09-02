#!/usr/bin/env python3
"""
Render the same calibrated video with multiple CRF values to compare quality.

Example:
python3 cap-com/test_video_crop_calibrator_crf.py \
  /path/to/input.mp4 \
  --config cap-com/crop_calibration_space_13.json \
  --crf-values 8 12 16 18 20 23 28 \
  --max-seconds 20 \
  --sample-seconds 10
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export one calibrated video at multiple CRF values for visual comparison.",
    )
    parser.add_argument("video", type=Path, help="Input video path")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Calibration JSON produced by video_crop_calibrator.py",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output folder. Default: next to input video",
    )
    parser.add_argument(
        "--crf-values",
        nargs="+",
        type=int,
        default=[8, 12, 16, 18, 20, 23, 28],
        help="CRF values to test. Lower is sharper/larger. Default: 8 12 16 18 20 23 28",
    )
    parser.add_argument(
        "--preset",
        default="slow",
        help="x264 preset for every export. Default: slow",
    )
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=15.0,
        help="Limit each rendered sample to N seconds. Default: 15",
    )
    parser.add_argument(
        "--sample-seconds",
        type=float,
        default=5.0,
        help="Extract one PNG frame at this time from each output. Default: 5",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing outputs.",
    )
    return parser.parse_args()


def ensure_ffmpeg_tools() -> tuple[str, str]:
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if not ffmpeg or not ffprobe:
        raise SystemExit("ffmpeg and ffprobe are required in PATH.")
    return ffmpeg, ffprobe


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def ffmpeg_filter_from_calib(c: dict) -> str:
    h = min(int(c["left_h"]), int(c["right_h"]))
    gain = clamp(float(c.get("right_luma_gain", 1.0)), 0.90, 1.10)

    left = (
        f"[0:v]crop={int(c['left_w'])}:{h}:"
        f"{int(c['left_x'])}:{int(c['left_y'])}[left]"
    )
    right = (
        f"[0:v]crop={int(c['right_w'])}:{h}:"
        f"{int(c['right_x'])}:{int(c['right_y'])}[right]"
    )

    if c.get("right_rotate_180", True):
        right_proc = (
            f"[right]hflip,vflip,"
            f"lutrgb=r='clip(val*{gain},0,255)':"
            f"g='clip(val*{gain},0,255)':"
            f"b='clip(val*{gain},0,255)'"
            f"[right_rot]"
        )
    else:
        right_proc = (
            f"[right]lutrgb=r='clip(val*{gain},0,255)':"
            f"g='clip(val*{gain},0,255)':"
            f"b='clip(val*{gain},0,255)'"
            f"[right_rot]"
        )

    gap_px = int(c.get("gap_px", 0))
    if gap_px > 0:
        stack = (
            f"color=c=white:s={gap_px}x{h}[gap];"
            f"[left][gap][right_rot]hstack=inputs=3[stacked]"
        )
    else:
        stack = "[left][right_rot]hstack=inputs=2[stacked]"

    even_pad = (
        "[stacked]"
        "pad=ceil(iw/2)*2:ceil(ih/2)*2:0:0:color=black"
        "[out]"
    )

    return ";".join([left, right, right_proc, stack, even_pad])


def run_command(command: list[str]) -> None:
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "command failed")


def probe_video(ffprobe: str, path: Path) -> dict:
    command = [
        ffprobe,
        "-v",
        "error",
        "-show_entries",
        "format=duration,size,bit_rate",
        "-of",
        "json",
        str(path),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "ffprobe failed")
    data = json.loads(result.stdout)
    return data.get("format", {})


def render_one(
    ffmpeg: str,
    input_video: Path,
    output_video: Path,
    filters: str,
    crf: int,
    preset: str,
    max_seconds: float | None,
    overwrite: bool,
) -> None:
    command = [
        ffmpeg,
        "-y" if overwrite else "-n",
        "-i",
        str(input_video),
        "-filter_complex",
        filters,
        "-map",
        "[out]",
        "-map",
        "0:a?",
        "-c:v",
        "libx264",
        "-crf",
        str(crf),
        "-preset",
        preset,
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "copy",
    ]
    if max_seconds is not None and max_seconds > 0:
        command += ["-t", str(max_seconds)]
    command.append(str(output_video))
    run_command(command)


def extract_sample_png(
    ffmpeg: str,
    input_video: Path,
    output_png: Path,
    sample_seconds: float,
    overwrite: bool,
) -> None:
    command = [
        ffmpeg,
        "-y" if overwrite else "-n",
        "-ss",
        str(sample_seconds),
        "-i",
        str(input_video),
        "-frames:v",
        "1",
        str(output_png),
    ]
    run_command(command)


def main() -> int:
    args = parse_args()
    ffmpeg, ffprobe = ensure_ffmpeg_tools()

    input_video = args.video.resolve()
    config_path = args.config.resolve()
    if not input_video.exists():
        raise SystemExit(f"Input video not found: {input_video}")
    if not config_path.exists():
        raise SystemExit(f"Config not found: {config_path}")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    filters = ffmpeg_filter_from_calib(config)

    if args.output_dir is None:
        output_dir = input_video.parent / f"{input_video.stem}_crf_test"
    else:
        output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    summary: list[dict] = []
    for crf in args.crf_values:
        output_video = output_dir / f"{input_video.stem}_crf{crf:02d}.mp4"
        output_png = output_dir / f"{input_video.stem}_crf{crf:02d}_sample.png"

        print(f"[CRF {crf}] rendering video -> {output_video.name}")
        render_one(
            ffmpeg=ffmpeg,
            input_video=input_video,
            output_video=output_video,
            filters=filters,
            crf=crf,
            preset=args.preset,
            max_seconds=args.max_seconds,
            overwrite=args.overwrite,
        )

        print(f"[CRF {crf}] extracting sample frame -> {output_png.name}")
        extract_sample_png(
            ffmpeg=ffmpeg,
            input_video=output_video,
            output_png=output_png,
            sample_seconds=args.sample_seconds,
            overwrite=args.overwrite,
        )

        info = probe_video(ffprobe, output_video)
        summary.append(
            {
                "crf": crf,
                "preset": args.preset,
                "video": str(output_video),
                "sample_png": str(output_png),
                "duration_seconds": float(info.get("duration", 0.0) or 0.0),
                "size_bytes": int(info.get("size", 0) or 0),
                "bit_rate": int(info.get("bit_rate", 0) or 0),
            }
        )

    summary_path = output_dir / "summary.json"
    note_path = output_dir / "note.txt"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    note_lines = [
        f"input_video: {input_video}",
        f"config: {config_path}",
        f"preset: {args.preset}",
        f"max_seconds: {args.max_seconds}",
        f"sample_seconds: {args.sample_seconds}",
        "",
        "CRF meaning:",
        "  lower CRF = better quality, larger file",
        "  higher CRF = softer/more compressed, smaller file",
        "",
        "Quick guide:",
        "  CRF 8-12  : very sharp, file gets big fast",
        "  CRF 16-18 : usually still good, balanced",
        "  CRF 20-23 : softness/compression starts to show",
        "  CRF 28+   : quality drop is usually obvious",
        "",
        "Open the PNG files side-by-side and compare fine detail, text edges, hair, and noise texture.",
    ]
    note_path.write_text("\n".join(note_lines) + "\n", encoding="utf-8")

    print(f"Saved: {summary_path}")
    print(f"Saved: {note_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
