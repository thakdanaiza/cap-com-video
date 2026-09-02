#!/usr/bin/env python3
"""
Build a slideshow video from the first and last frame of each clip.

The script scans clip files in timestamp order, captures two images per clip
(`first` then `last`), and renders them into one slideshow video while also
writing note.txt and manifest.json metadata.

Example:
python3 build_clip_edge_frames.py \
  --input-path /path/to/flip/ch3-sleep \
  --frames-per-image 60 \
  --fps 60 \
  --output ./_edge_frame_exports
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable

from build_long_video_from_clip_ranges import RangeRow

FILENAME_RE = re.compile(
    r"^double_(?P<date>\d{8})_(?P<time>\d{6})_(?P<offset>\d{3,})\.mp4$",
    re.IGNORECASE,
)
DEFAULT_SOURCE_ROOT = Path(os.environ.get("CAP_COM_FLIP", "flip"))
LAST_FRAME_LOOKBACK_SECONDS = 0.5
DEFAULT_SOURCE_MODE = "video-fps"


@dataclass(frozen=True)
class Clip:
    path: Path
    relative_path: str
    base_start: datetime
    offset_minutes: int
    clip_start: datetime


@dataclass(frozen=True)
class FrameItem:
    index: int
    clip: Clip
    role: str
    clip_duration_seconds: float
    capture_second: float
    capture_time: datetime
    image_path: Path
    output_start_seconds: float
    output_end_seconds: float


@dataclass(frozen=True)
class EdgeFrameJob:
    row_index: int
    row: RangeRow
    clips: list[Clip]
    skipped_missing: list[str]


def log(message: str) -> None:
    print(f"[build_clip_edge_frames] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture first/last frames from clips and build a slideshow video.",
    )
    parser.add_argument(
        "--source-mode",
        choices=("edge-images", "video-fps"),
        default=DEFAULT_SOURCE_MODE,
        help="Build mode. 'edge-images' keeps first/last frame behavior, 'video-fps' samples frames directly from video. Default: video-fps",
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=DEFAULT_SOURCE_ROOT,
        help=f"Root folder to scan when --input-path is omitted. Default: {DEFAULT_SOURCE_ROOT}",
    )
    parser.add_argument(
        "--input-path",
        dest="input_paths",
        action="append",
        type=Path,
        default=[],
        help=(
            "Optional specific file or folder to include. "
            "Use multiple times to combine many files/folders. "
            "If omitted, the script scans --source-root."
        ),
    )
    parser.add_argument(
        "--output",
        "--output-root",
        "--output-dir",
        dest="output_root",
        type=Path,
        default=None,
        help="Parent folder where the auto-named output folder will be created. Default: inside the input folder.",
    )
    parser.add_argument(
        "--output-name",
        default=None,
        help="Optional base name for the output folder/video.",
    )
    parser.add_argument(
        "--frame-hold-seconds",
        type=float,
        default=None,
        help="How long each captured image stays on screen. If omitted, uses --frames-per-image / --fps.",
    )
    parser.add_argument(
        "--frames-per-image",
        "--image-fps",
        dest="frames_per_image",
        type=float,
        default=1.0,
        help=(
            "How many output frames each captured image should stay on screen. "
            "Example: --frames-per-image 60 --fps 60 means 1 second per image. Default: 1"
        ),
    )
    parser.add_argument("--fps", type=int, default=30, help="Rendered slideshow video fps. Default: 30")
    parser.add_argument(
        "--sample-fps",
        type=float,
        default=1.0,
        help="When --source-mode=video-fps, sample this many frames per source second. Default: 1",
    )
    parser.add_argument(
        "--default-clip-seconds",
        type=float,
        default=60.0,
        help="Fallback clip duration when ffprobe is unavailable. Default: 60",
    )
    parser.add_argument(
        "--jpg-quality",
        type=int,
        default=2,
        help="FFmpeg MJPEG quality for extracted frames. Lower is better. Default: 2",
    )
    parser.add_argument(
        "--crf",
        type=int,
        default=0,
        help="libx264 CRF quality for slideshow video. Lower is better. 0 is lossless. Default: 0",
    )
    parser.add_argument(
        "--preset",
        default="veryslow",
        help="libx264 preset. Default: veryslow",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview selected clips and output plan without writing files.",
    )
    return parser.parse_args()


def normalize_timing(
    frame_hold_seconds: float | None,
    frames_per_image: float,
    render_fps: int,
) -> tuple[float, float, float]:
    if render_fps <= 0:
        raise ValueError("--fps must be greater than 0.")
    if frame_hold_seconds is not None:
        if frame_hold_seconds <= 0:
            raise ValueError("--frame-hold-seconds must be greater than 0.")
        return frame_hold_seconds, frame_hold_seconds * render_fps, 1.0 / frame_hold_seconds
    if frames_per_image <= 0:
        raise ValueError("--frames-per-image must be greater than 0.")
    frame_hold = frames_per_image / render_fps
    return frame_hold, frames_per_image, 1.0 / frame_hold


def normalize_sample_fps(sample_fps: float) -> float:
    if sample_fps <= 0:
        raise ValueError("--sample-fps must be greater than 0.")
    return sample_fps


def parse_clip(path: Path, source_root: Path) -> Clip | None:
    match = FILENAME_RE.match(path.name)
    if not match:
        return None
    base_start = datetime.strptime(
        f"{match.group('date')}_{match.group('time')}",
        "%Y%m%d_%H%M%S",
    )
    offset_minutes = int(match.group("offset"))
    clip_start = base_start + timedelta(minutes=offset_minutes)
    return Clip(
        path=path,
        relative_path=path.relative_to(source_root).as_posix(),
        base_start=base_start,
        offset_minutes=offset_minutes,
        clip_start=clip_start,
    )


def discover_clips_from_root(source_root: Path) -> list[Clip]:
    log(f"Scanning folder: {source_root}")
    clips: list[Clip] = []
    for index, path in enumerate(sorted(source_root.rglob("*.mp4")), start=1):
        if "_long_video_exports" in path.parts or "_edge_frame_exports" in path.parts:
            continue
        clip = parse_clip(path, source_root)
        if clip is not None:
            clips.append(clip)
            if index % 100 == 0:
                log(f"Scanned {index} files, matched {len(clips)} clips")
    log(f"Finished scan: found {len(clips)} usable clips")
    return sorted(clips, key=lambda item: (item.clip_start, item.relative_path))


def discover_clips_from_inputs(input_paths: list[Path]) -> list[Clip]:
    clips_by_path: dict[Path, Clip] = {}
    for input_path in input_paths:
        resolved = input_path.resolve()
        if not resolved.exists():
            raise ValueError(f"Input path was not found: {resolved}")

        if resolved.is_dir():
            log(f"Reading input folder: {resolved}")
            discovered = discover_clips_from_root(resolved)
        else:
            log(f"Reading input file: {resolved}")
            if resolved.suffix.lower() != ".mp4":
                continue
            clip = parse_clip(resolved, resolved.parent)
            discovered = [clip] if clip is not None else []

        for clip in discovered:
            clips_by_path[clip.path.resolve()] = clip

    log(f"Collected {len(clips_by_path)} unique clips from explicit input paths")
    return sorted(clips_by_path.values(), key=lambda item: (item.clip_start, item.relative_path))


def build_clip_index_from_paths(source_root: Path, paths, label: str = "provided paths") -> dict[tuple[str, int], Clip]:
    source_root = source_root.resolve()
    log(f"Building clip index from {label}")
    index: dict[tuple[str, int], Clip] = {}
    scanned = 0
    for path in sorted({Path(path).expanduser().resolve() for path in paths}):
        if path.suffix.lower() != ".mp4":
            continue
        try:
            path.relative_to(source_root)
        except ValueError:
            continue
        scanned += 1
        if "_long_video_exports" in path.parts or "_edge_frame_exports" in path.parts:
            continue
        clip = parse_clip(path, source_root)
        if clip is None:
            continue
        clip_set = clip.base_start.strftime("%Y%m%d_%H%M%S")
        index[(clip_set, clip.offset_minutes)] = clip
        if scanned % 200 == 0:
            log(f"Scanned {scanned} files, indexed {len(index)} clips")
    log(f"Finished clip index: scanned {scanned} mp4 files, indexed {len(index)} clips")
    return index


def build_clip_index(source_root: Path) -> dict[tuple[str, int], Clip]:
    source_root = source_root.resolve()
    log(f"Scanning clip index under: {source_root}")
    return build_clip_index_from_paths(source_root, source_root.rglob("*.mp4"), "source root scan")


def collect_jobs_and_missing(
    rows: list[RangeRow],
    source_root: Path,
    clip_index: dict[tuple[str, int], Clip],
) -> tuple[list[EdgeFrameJob], list[str], list[str]]:
    jobs: list[EdgeFrameJob] = []
    missing: list[str] = []
    skipped_missing: list[str] = []
    for row in rows:
        clips: list[Clip] = []
        row_skipped_missing: list[str] = []
        step = 1 if row.stop >= row.start else -1
        for offset in range(row.start, row.stop + step, step):
            clip = clip_index.get((row.clip_set, offset))
            if clip is None:
                line = f"row {row.row_number:03d}: double_{row.clip_set}_{offset:03d}.mp4"
                row_skipped_missing.append(line)
                skipped_missing.append(line)
                continue
            clips.append(clip)
        if clips:
            jobs.append(
                EdgeFrameJob(
                    row_index=row.row_number,
                    row=row,
                    clips=clips,
                    skipped_missing=row_skipped_missing,
                )
            )
        elif row_skipped_missing:
            missing.extend(row_skipped_missing)
    return jobs, missing, skipped_missing


def probe_duration_seconds(path: Path, fallback_seconds: float) -> float:
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return fallback_seconds
    result = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return fallback_seconds
    try:
        duration = float(result.stdout.strip())
    except ValueError:
        return fallback_seconds
    return duration if duration > 0 else fallback_seconds


def probe_dimensions(path: Path) -> tuple[int, int] | None:
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return None
    result = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "csv=p=0:s=x",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    text = result.stdout.strip()
    if "x" not in text:
        return None
    width_text, height_text = text.split("x", 1)
    try:
        width = int(width_text)
        height = int(height_text)
    except ValueError:
        return None
    if width <= 0 or height <= 0:
        return None
    return width, height


def ensure_tool(name: str) -> str:
    path = shutil.which(name)
    if not path:
        raise RuntimeError(f"{name} was not found in PATH.")
    return path


def format_seconds(total_seconds: float) -> str:
    rounded = int(total_seconds)
    hours, remainder = divmod(rounded, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def build_output_name(clips: list[Clip], requested_name: str | None) -> str:
    if requested_name:
        return requested_name
    if not clips:
        return "clip_edge_frames"
    start_at = clips[0].clip_start
    end_at = clips[-1].clip_start
    return f"edge_frames_{start_at:%Y%m%d_%H%M%S}__{end_at:%Y%m%d_%H%M%S}"


def row_folder_name(job: EdgeFrameJob) -> str:
    return f"row_{job.row_index:03d}_{job.row.clip_set}_{job.row.start:03d}_{job.row.stop:03d}_edge_frames"


def resolve_output_root_for_job(job: EdgeFrameJob, output_root: Path | None) -> Path:
    if output_root is not None:
        return output_root

    clip_path = job.clips[0].path
    for parent in [clip_path.parent, *clip_path.parents]:
        if parent.name in {"step", "flip"}:
            return parent.parent / "event"
    return clip_path.parent / "event"


def build_output_paths_for_job(output_root: Path | None, job: EdgeFrameJob) -> tuple[Path, Path, Path, Path, Path]:
    resolved_output_root = resolve_output_root_for_job(job, output_root)
    folder_name = row_folder_name(job)
    output_dir = resolved_output_root / folder_name
    return (
        output_dir,
        output_dir / "frames",
        output_dir / f"{folder_name}.mp4",
        output_dir / "note.txt",
        output_dir / "manifest.json",
    )


def resolve_output_root(args: argparse.Namespace, clips: list[Clip]) -> Path:
    if args.output_root is not None:
        return args.output_root.resolve()
    if len(args.input_paths) == 1:
        input_path = args.input_paths[0].resolve()
        if input_path.is_dir():
            return input_path
        return input_path.parent
    if clips:
        return clips[0].path.parent
    return Path.cwd()


def write_outputs(
    output_dir: Path,
    images_dir: Path,
    output_video: Path,
    note_path: Path,
    manifest_path: Path,
    clips: list[Clip],
    frame_items: list[FrameItem],
    durations_by_path: dict[Path, float],
    frame_hold_seconds: float,
    frames_per_image: float,
    images_per_second: float,
    render_fps: int,
    jpg_quality: int,
    crf: int,
    preset: str,
    source_mode: str,
    sample_fps: float,
    progress_callback=None,
) -> None:
    ffmpeg = ensure_tool("ffmpeg")
    output_dir.mkdir(parents=True, exist_ok=True)
    if source_mode == "edge-images":
        images_dir.mkdir(parents=True, exist_ok=True)
        capture_frames(ffmpeg, frame_items, jpg_quality, progress_callback=progress_callback)
        concat_file = output_dir / "concat_frames.txt"
        write_concat_file(frame_items, frame_hold_seconds, concat_file)
        if progress_callback is not None:
            progress_callback(
                {
                    "phase": "concat",
                    "output_video": str(output_video),
                }
            )
        build_slideshow_video(ffmpeg, concat_file, output_video, render_fps, crf, preset)
        write_note(
            note_path,
            output_video,
            clips,
            frame_items,
            frame_hold_seconds,
            frames_per_image,
            images_per_second,
            render_fps,
            crf,
            preset,
            source_mode,
            sample_fps,
            durations_by_path,
        )
        manifest = build_manifest(
            output_video=output_video,
            clips=clips,
            frame_items=frame_items,
            durations_by_path=durations_by_path,
            frame_hold_seconds=frame_hold_seconds,
            frames_per_image=frames_per_image,
            images_per_second=images_per_second,
            render_fps=render_fps,
            crf=crf,
            preset=preset,
            source_mode=source_mode,
            sample_fps=sample_fps,
        )
    else:
        segment_paths = render_video_edge_segments(
            ffmpeg=ffmpeg,
            clips=clips,
            durations_by_path=durations_by_path,
            output_dir=output_dir,
            frames_per_image=frames_per_image,
            render_fps=render_fps,
            crf=crf,
            preset=preset,
            progress_callback=progress_callback,
        )
        concat_segments(segment_paths, output_video, ffmpeg)
        write_note(
            note_path,
            output_video,
            clips,
            [],
            1.0 / render_fps,
            frames_per_image,
            sample_fps,
            render_fps,
            crf,
            preset,
            source_mode,
            sample_fps,
            durations_by_path,
        )
        manifest = build_manifest(
            output_video=output_video,
            clips=clips,
            frame_items=[],
            durations_by_path=durations_by_path,
            frame_hold_seconds=1.0 / render_fps,
            frames_per_image=frames_per_image,
            images_per_second=sample_fps,
            render_fps=render_fps,
            crf=crf,
            preset=preset,
            source_mode=source_mode,
            sample_fps=sample_fps,
        )
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def render_video_edge_segments(
    ffmpeg: str,
    clips: list[Clip],
    durations_by_path: dict[Path, float],
    output_dir: Path,
    frames_per_image: float,
    render_fps: int,
    crf: int,
    preset: str,
    progress_callback=None,
) -> list[Path]:
    segments_dir = output_dir / "segments"
    segments_dir.mkdir(parents=True, exist_ok=True)
    target_size = probe_dimensions(clips[0].path)
    rendered: list[Path] = []
    total = len(clips)
    segment_frames = max(1, int(round(frames_per_image)))
    edge_seconds = segment_frames / max(render_fps, 1)
    for index, clip in enumerate(clips, start=1):
        if progress_callback is not None:
            progress_callback(
                {
                    "phase": "rendering",
                    "current_clip": index,
                    "total_clips": total,
                    "current_file": f"{clip.relative_path} [video-fps]",
                }
            )
        log(
            f"Sampling clip {index}/{total}: {clip.relative_path} "
            f"[head {segment_frames}f + tail {segment_frames}f]"
        )
        for role, start_from_end in (("head", False), ("tail", True)):
            segment_path = segments_dir / f"segment_{index:04d}_{role}.mp4"
            filters = []
            if target_size is not None:
                width, height = target_size
                filters.extend(
                    [
                        f"scale={width}:{height}:force_original_aspect_ratio=decrease",
                        f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2",
                        "setsar=1",
                    ]
                )
            filters.append(f"setpts=N/({render_fps}*TB)")
            command = [ffmpeg, "-y"]
            if start_from_end:
                seek_seconds = min(durations_by_path[clip.path], max(edge_seconds, LAST_FRAME_LOOKBACK_SECONDS))
                command.extend(["-sseof", f"-{seek_seconds:.3f}"])
            command.extend(
                [
                    "-i",
                    str(clip.path),
                    "-an",
                    "-frames:v",
                    str(segment_frames),
                    "-vf",
                    ",".join(filters),
                    "-c:v",
                    "libx264",
                    "-preset",
                    preset,
                    "-crf",
                    str(crf),
                    "-pix_fmt",
                    "yuv444p",
                    "-movflags",
                    "+faststart",
                    str(segment_path),
                ]
            )
            result = subprocess.run(command, capture_output=True, text=True, check=False)
            if result.returncode != 0:
                raise RuntimeError(
                    f"ffmpeg failed while sampling {role} frames from {clip.relative_path}\n"
                    f"{result.stderr.strip()}",
                )
            rendered.append(segment_path)
    return rendered


def concat_segments(segment_paths: list[Path], output_video: Path, ffmpeg: str) -> None:
    concat_file = output_video.parent / "concat_segments.txt"
    concat_lines = [f"file '{segment.as_posix()}'" for segment in segment_paths]
    concat_file.write_text("\n".join(concat_lines) + "\n", encoding="utf-8")
    if not segment_paths:
        raise RuntimeError("No segments were rendered.")
    result = subprocess.run(
        [
            ffmpeg,
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_file),
            "-c",
            "copy",
            str(output_video),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed while concatenating video-fps segments\n{result.stderr.strip()}")


def build_frame_items(
    clips: Iterable[Clip],
    images_dir: Path,
    durations_by_path: dict[Path, float],
    frame_hold_seconds: float,
) -> list[FrameItem]:
    items: list[FrameItem] = []
    cursor = 0.0
    index = 1
    for clip in clips:
        duration_seconds = durations_by_path[clip.path]
        last_second = max(duration_seconds - LAST_FRAME_LOOKBACK_SECONDS, 0.0)
        roles = (
            ("first", 0.0),
            ("last", last_second),
        )
        for role, capture_second in roles:
            capture_time = clip.clip_start + timedelta(seconds=capture_second)
            timestamp_label = capture_time.strftime("%Y%m%d_%H%M%S")
            image_path = images_dir / f"{index:04d}_{timestamp_label}_{role}.jpg"
            items.append(
                FrameItem(
                    index=index,
                    clip=clip,
                    role=role,
                    clip_duration_seconds=duration_seconds,
                    capture_second=capture_second,
                    capture_time=capture_time,
                    image_path=image_path,
                    output_start_seconds=cursor,
                    output_end_seconds=cursor + frame_hold_seconds,
                )
            )
            cursor += frame_hold_seconds
            index += 1
    return items


def capture_frames(ffmpeg: str, frame_items: list[FrameItem], jpg_quality: int, progress_callback=None) -> None:
    total = len(frame_items)
    for item in frame_items:
        if progress_callback is not None:
            progress_callback(
                {
                    "phase": "rendering",
                    "current_clip": item.index,
                    "total_clips": total,
                    "current_file": f"{item.clip.relative_path} [{item.role}]",
                }
            )
        log(
            f"Capturing frame {item.index}/{total}: "
            f"{item.clip.relative_path} [{item.role}]"
        )
        commands: list[list[str]] = [
            [
                ffmpeg,
                "-y",
                "-i",
                str(item.clip.path),
                "-ss",
                f"{item.capture_second:.3f}",
                "-frames:v",
                "1",
                "-q:v",
                str(jpg_quality),
                str(item.image_path),
            ]
        ]
        if item.role == "last":
            seek_from_end = min(LAST_FRAME_LOOKBACK_SECONDS, max(item.clip_duration_seconds, 0.0))
            commands.append(
                [
                    ffmpeg,
                    "-y",
                    "-sseof",
                    f"-{seek_from_end:.3f}",
                    "-i",
                    str(item.clip.path),
                    "-frames:v",
                    "1",
                    "-q:v",
                    str(jpg_quality),
                    str(item.image_path),
                ]
            )

        last_error = ""
        for command in commands:
            if item.image_path.exists():
                item.image_path.unlink()
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode == 0 and item.image_path.exists() and item.image_path.stat().st_size > 0:
                break
            last_error = result.stderr.strip()
        else:
            raise RuntimeError(
                f"ffmpeg failed while capturing {item.role} frame from {item.clip.relative_path}\n"
                f"{last_error or 'frame image was not created'}"
            )


def write_concat_file(frame_items: list[FrameItem], frame_hold_seconds: float, concat_file: Path) -> None:
    lines: list[str] = []
    for item in frame_items:
        lines.append(f"file '{item.image_path.as_posix()}'")
        lines.append(f"duration {frame_hold_seconds:.3f}")
    if frame_items:
        lines.append(f"file '{frame_items[-1].image_path.as_posix()}'")
    concat_file.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_slideshow_video(ffmpeg: str, concat_file: Path, output_video: Path, fps: int, crf: int, preset: str) -> None:
    result = subprocess.run(
        [
            ffmpeg,
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_file),
            "-fps_mode",
            "vfr",
            "-vf",
            f"fps={fps}",
            "-c:v",
            "libx264",
            "-preset",
            preset,
            "-crf",
            str(crf),
            "-pix_fmt",
            "yuv444p",
            "-movflags",
            "+faststart",
            str(output_video),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed while building slideshow video\n{result.stderr.strip()}")


def write_note(
    note_path: Path,
    output_video: Path,
    clips: list[Clip],
    frame_items: list[FrameItem],
    frame_hold_seconds: float,
    frames_per_image: float,
    images_per_second: float,
    render_fps: int,
    crf: int,
    preset: str,
    source_mode: str,
    sample_fps: float,
    durations_by_path: dict[Path, float],
) -> None:
    lines = [
        f"output_video: {output_video.name}",
        f"source_mode: {source_mode}",
        f"source_clip_count: {len(clips)}",
        f"frames: {len(frame_items)}",
        f"frame_hold_seconds: {frame_hold_seconds}",
        f"frames_per_image: {frames_per_image}",
        f"images_per_second: {images_per_second}",
        f"render_fps: {render_fps}",
        f"sample_fps: {sample_fps}",
        f"crf: {crf}",
        f"preset: {preset}",
        "",
        "timeline:",
    ]
    if source_mode == "edge-images":
        for item in frame_items:
            lines.append(
                " | ".join(
                    [
                        f"{item.index:04d}",
                        f"output {format_seconds(item.output_start_seconds)} -> {format_seconds(item.output_end_seconds)}",
                        f"capture_time {item.capture_time.isoformat(sep=' ')}",
                        f"capture_second {item.capture_second:.3f}s",
                        f"role {item.role}",
                        f"file {item.clip.relative_path}",
                    ]
                )
            )
    else:
        cursor = 0.0
        for index, clip in enumerate(clips, start=1):
            duration_seconds = durations_by_path[clip.path]
            sampled_frames = max(1, int(round(frames_per_image))) * 2
            output_seconds = sampled_frames / render_fps
            lines.append(
                " | ".join(
                    [
                        f"{index:04d}",
                        f"output {format_seconds(cursor)} -> {format_seconds(cursor + output_seconds)}",
                        f"clip_start {clip.clip_start.isoformat(sep=' ')}",
                        f"source_seconds {duration_seconds:.3f}",
                        f"sampled_frames {sampled_frames}",
                        f"file {clip.relative_path}",
                    ]
                )
            )
            cursor += output_seconds
    note_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_manifest(
    output_video: Path,
    clips: list[Clip],
    frame_items: list[FrameItem],
    durations_by_path: dict[Path, float],
    frame_hold_seconds: float,
    frames_per_image: float,
    images_per_second: float,
    render_fps: int,
    crf: int,
    preset: str,
    source_mode: str,
    sample_fps: float,
) -> dict:
    manifest = {
        "output_video": output_video.name,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source_mode": source_mode,
        "source_clip_count": len(clips),
        "frames_count": len(frame_items),
        "frame_hold_seconds": frame_hold_seconds,
        "frames_per_image": frames_per_image,
        "images_per_second": images_per_second,
        "render_fps": render_fps,
        "sample_fps": sample_fps,
        "crf": crf,
        "preset": preset,
    }
    if source_mode == "edge-images":
        manifest["segments"] = [
            {
                "index": item.index,
                "source_file": item.clip.relative_path,
                "source_path": str(item.clip.path),
                "base_start": item.clip.base_start.isoformat(sep=" "),
                "offset_minutes": item.clip.offset_minutes,
                "clip_start": item.clip.clip_start.isoformat(sep=" "),
                "clip_duration_seconds": round(durations_by_path[item.clip.path], 3),
                "frame_role": item.role,
                "capture_second": round(item.capture_second, 3),
                "capture_time": item.capture_time.isoformat(sep=" "),
                "source_start": item.capture_time.isoformat(sep=" "),
                "source_end": item.capture_time.isoformat(sep=" "),
                "image_file": item.image_path.name,
                "output_start_seconds": round(item.output_start_seconds, 3),
                "output_end_seconds": round(item.output_end_seconds, 3),
            }
            for item in frame_items
        ]
    else:
        manifest["segments"] = [
            {
                "index": index,
                "source_file": clip.relative_path,
                "source_path": str(clip.path),
                "base_start": clip.base_start.isoformat(sep=" "),
                "offset_minutes": clip.offset_minutes,
                "clip_start": clip.clip_start.isoformat(sep=" "),
                "clip_duration_seconds": round(durations_by_path[clip.path], 3),
                "sample_fps": sample_fps,
                "head_frames": max(1, int(round(frames_per_image))),
                "tail_frames": max(1, int(round(frames_per_image))),
                "estimated_sampled_frames": max(1, int(round(frames_per_image))) * 2,
            }
            for index, clip in enumerate(clips, start=1)
        ]
    return manifest


def build_edge_frame_videos_from_rows(
    rows: list[RangeRow],
    source_root: Path,
    output_root: Path | None = None,
    frame_hold_seconds: float | None = None,
    frames_per_image: float = 1.0,
    render_fps: int = 30,
    default_clip_seconds: float = 60.0,
    jpg_quality: int = 2,
    crf: int = 0,
    preset: str = "veryslow",
    source_mode: str = DEFAULT_SOURCE_MODE,
    sample_fps: float = 1.0,
    dry_run: bool = False,
    progress_callback=None,
    clip_index: dict[tuple[str, int], Clip] | None = None,
    allow_missing: bool = True,
) -> dict:
    source_root = source_root.resolve()
    normalized_sample_fps = normalize_sample_fps(sample_fps)
    normalized_hold, normalized_frames_per_image, images_per_second = normalize_timing(
        frame_hold_seconds,
        frames_per_image,
        render_fps,
    )
    clip_index = clip_index or build_clip_index(source_root)
    jobs, missing, skipped_missing = collect_jobs_and_missing(rows, source_root, clip_index)
    if not allow_missing and skipped_missing:
        missing.extend(skipped_missing)
    if missing:
        raise RuntimeError(
            "Missing clips referenced by ranges file:\n"
            + "\n".join(missing[:30])
            + (f"\n... and {len(missing) - 30} more missing clips" if len(missing) > 30 else "")
        )
    if not jobs:
        raise RuntimeError("No clips were selected from the provided ranges.")

    results = {
        "rows": len(rows),
        "jobs": len(jobs),
        "outputs": [],
        "missing": skipped_missing,
    }
    for job_index, job in enumerate(jobs, start=1):
        output_dir, images_dir, output_video, note_path, manifest_path = build_output_paths_for_job(output_root, job)
        durations_by_path = {
            clip.path: probe_duration_seconds(clip.path, default_clip_seconds)
            for clip in job.clips
        }
        frame_items = build_frame_items(job.clips, images_dir, durations_by_path, normalized_hold)
        item_result = {
            "row_index": job.row_index,
            "clip_set": job.row.clip_set,
            "start": job.row.start,
            "stop": job.row.stop,
            "output_video": str(output_video),
            "note_path": str(note_path),
            "manifest_path": str(manifest_path),
            "frames_dir": str(images_dir),
            "source_clip_count": len(job.clips),
            "frames_count": len(frame_items),
            "skipped_missing": job.skipped_missing,
            "crf": crf,
            "preset": preset,
        }
        if dry_run:
            results["outputs"].append(item_result)
            continue

        if progress_callback is not None:
            progress_callback(
                {
                    "phase": "row_start",
                    "job_index": job_index,
                    "total_jobs": len(jobs),
                    "row_index": job.row_index,
                    "clip_set": job.row.clip_set,
                    "start": job.row.start,
                    "stop": job.row.stop,
                    "output_dir": str(output_dir),
                    "total_clips": len(job.clips),
                }
            )
        try:
            write_outputs(
                output_dir=output_dir,
                images_dir=images_dir,
                output_video=output_video,
                note_path=note_path,
                manifest_path=manifest_path,
                clips=job.clips,
                frame_items=frame_items,
                durations_by_path=durations_by_path,
                frame_hold_seconds=normalized_hold,
                frames_per_image=normalized_frames_per_image,
                images_per_second=images_per_second,
                render_fps=render_fps,
                jpg_quality=jpg_quality,
                crf=crf,
                preset=preset,
                source_mode=source_mode,
                sample_fps=normalized_sample_fps,
                progress_callback=progress_callback,
            )
        except RuntimeError as exc:
            raise RuntimeError(
                f"row {job.row_index:03d} failed for {job.row.clip_set} {job.row.start}->{job.row.stop}: {exc}"
            ) from exc
        results["outputs"].append(item_result)
        if progress_callback is not None:
            progress_callback(
                {
                    "phase": "row_done",
                    "job_index": job_index,
                    "total_jobs": len(jobs),
                    **item_result,
                }
            )
    return results


def main() -> int:
    args = parse_args()
    try:
        sample_fps = normalize_sample_fps(args.sample_fps)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    try:
        frame_hold_seconds, frames_per_image, images_per_second = normalize_timing(
            args.frame_hold_seconds,
            args.frames_per_image,
            args.fps,
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    try:
        if args.input_paths:
            log(f"Using {len(args.input_paths)} explicit input path(s)")
            clips = discover_clips_from_inputs(args.input_paths)
        else:
            log(f"Using source root: {args.source_root.resolve()}")
            clips = discover_clips_from_root(args.source_root.resolve())
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    if not clips:
        print("No matching video files were found.", file=sys.stderr)
        return 1

    durations_by_path = {
        clip.path: probe_duration_seconds(clip.path, args.default_clip_seconds)
        for clip in clips
    }

    output_root = resolve_output_root(args, clips)
    base_name = build_output_name(clips, args.output_name)
    output_dir = output_root / base_name
    images_dir = output_dir / "frames"
    output_video = output_dir / f"{base_name}.mp4"
    note_path = output_dir / "note.txt"
    manifest_path = output_dir / "manifest.json"

    frame_items = build_frame_items(clips, images_dir, durations_by_path, frame_hold_seconds)

    if args.dry_run:
        log("Dry-run mode: no files will be written")
        print(f"Clips        : {len(clips)}")
        print(f"Frames       : {len(frame_items)}")
        print(f"Output dir   : {output_dir}")
        print(f"Output video : {output_video}")
        print(f"Frames/image : {frames_per_image}")
        print(f"Images/sec   : {images_per_second}")
        print(f"Hold seconds : {frame_hold_seconds}")
        print(f"Render fps   : {args.fps}")
        print(f"Source mode  : {args.source_mode}")
        print(f"Sample fps   : {sample_fps}")
        print(f"CRF          : {args.crf}")
        print(f"Preset       : {args.preset}")
        print(f"Total length : {len(frame_items) * frame_hold_seconds:.3f}s")
        return 0

    try:
        write_outputs(
            output_dir=output_dir,
            images_dir=images_dir,
            output_video=output_video,
            note_path=note_path,
            manifest_path=manifest_path,
            clips=clips,
            frame_items=frame_items,
            durations_by_path=durations_by_path,
            frame_hold_seconds=frame_hold_seconds,
            frames_per_image=frames_per_image,
            images_per_second=images_per_second,
            render_fps=args.fps,
            jpg_quality=args.jpg_quality,
            crf=args.crf,
            preset=args.preset,
            source_mode=args.source_mode,
            sample_fps=sample_fps,
        )
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(f"Created : {output_video}")
    print(f"Frames  : {images_dir}")
    print(f"Note    : {note_path}")
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
