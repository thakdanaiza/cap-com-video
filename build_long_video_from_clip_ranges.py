#!/usr/bin/env python3
"""
Build one output video per row from clip_set/start/stop ranges.

TSV example:
clip_set	start	stop
20260527_153639	0	12
20260527_153639	15	15
20260527_153639	0	57

Each row becomes one output video.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path


DEFAULT_SOURCE_ROOT = Path(os.environ.get("CAP_COM_FLIP", "flip"))
DEFAULT_OUTPUT_ROOT = Path(os.environ.get("CAP_COM_OUTPUT_ROOT", "outputs"))


@dataclass(frozen=True)
class RangeRow:
    row_number: int
    clip_set: str
    start: int
    stop: int


@dataclass(frozen=True)
class SequenceEntry:
    kind: str
    clip_set: str = ""
    start: int = 0
    stop: int = 0
    file_ref: str = ""


@dataclass(frozen=True)
class SequenceRow:
    row_number: int
    entries: list[SequenceEntry]
    output_name: str = ""


@dataclass(frozen=True)
class ClipItem:
    row_index: int
    clip_set: str
    offset: int
    path: Path
    relative_path: str
    clip_start: datetime


@dataclass(frozen=True)
class RowJob:
    row_index: int
    row: RangeRow | SequenceRow
    items: list[ClipItem]
    skipped_missing: list[str]


@dataclass(frozen=True)
class ClipCatalog:
    by_key: dict[tuple[str, int], Path]
    by_rel_path: dict[str, Path]
    by_name: dict[str, list[Path]]
    by_dir_rel_path: dict[str, list[Path]]
    by_dir_name: dict[str, list[Path]]


def log(message: str) -> None:
    print(f"[build_clip_ranges] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create one video per clip_set/start/stop row.")
    parser.add_argument("--ranges-file", type=Path, required=True, help="TSV/CSV file with clip_set,start,stop")
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT, help=f"Default: {DEFAULT_SOURCE_ROOT}")
    parser.add_argument(
        "--output",
        "--output-dir",
        "--output-root",
        dest="output_root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Parent folder for generated row folders.",
    )
    parser.add_argument("--crf", type=int, default=0, help="libx264 CRF quality. 0 is lossless. Default: 0")
    parser.add_argument("--preset", default="veryslow", help="libx264 preset. Default: veryslow")
    parser.add_argument("--default-clip-seconds", type=float, default=60.0, help="Fallback duration. Default: 60")
    parser.add_argument("--dry-run", action="store_true", help="Preview only")
    return parser.parse_args()


def detect_delimiter(text: str) -> str:
    first_line = next((line for line in text.splitlines() if line.strip()), "")
    if "," in first_line and "\t" not in first_line:
        return ","
    return "\t"


def parse_clip_set_datetime(clip_set: str) -> datetime:
    return datetime.strptime(clip_set, "%Y%m%d_%H%M%S")


def parse_clip_file_name(name: str) -> tuple[str, int] | None:
    stem = Path(name).name
    if not stem.lower().endswith(".mp4"):
        return None
    parts = Path(stem).stem.split("_")
    if len(parts) < 4 or parts[0] != "double":
        return None
    clip_set = f"{parts[1]}_{parts[2]}"
    try:
        offset = int(parts[3])
    except ValueError:
        return None
    return clip_set, offset


def reference_lookup_keys(raw_value: str) -> list[str]:
    raw = str(raw_value or "").replace("\\", "/").strip()
    if not raw:
        return []

    variants: list[str] = []

    def add(value: str) -> None:
        cleaned = value.strip().strip("/")
        while cleaned.startswith("./"):
            cleaned = cleaned[2:]
        if cleaned and cleaned not in variants:
            variants.append(cleaned)

    add(raw)

    parts = [part for part in raw.split("/") if part not in {"", "."}]
    for marker in ("flip", "non-flip"):
        if marker in parts:
            index = parts.index(marker)
            suffix = "/".join(parts[index + 1 :])
            add(suffix)
            add("/".join(parts[index:]))

    if len(parts) > 1:
        add("/".join(parts[1:]))

    return variants


def resolve_file_reference(file_ref: str, clip_index: ClipCatalog) -> tuple[str, int, Path | None, str]:
    raw = str(file_ref or "").strip()
    if not raw:
        return "", 0, None, "(empty file reference)"

    parsed = parse_clip_file_name(raw)
    if parsed is not None:
        clip_set, offset = parsed
        return clip_set, offset, clip_index.by_key.get((clip_set, offset)), f"double_{clip_set}_{offset:03d}.mp4"

    for rel_key in reference_lookup_keys(raw):
        path = clip_index.by_rel_path.get(rel_key)
        if path is not None:
            parsed = parse_clip_file_name(path.name)
            if parsed is not None:
                clip_set, offset = parsed
                return clip_set, offset, path, rel_key
            return "", 0, path, rel_key

    matches = clip_index.by_name.get(Path(raw).name, [])
    if len(matches) == 1:
        path = matches[0]
        parsed = parse_clip_file_name(path.name)
        if parsed is not None:
            clip_set, offset = parsed
            return clip_set, offset, path, path.name
        return "", 0, path, path.name
    if len(matches) > 1:
        return "", 0, None, f"{raw} (matched multiple files; use relative path)"
    return "", 0, None, raw


def resolve_folder_reference(folder_ref: str, clip_index: ClipCatalog) -> list[tuple[str, int, Path | None, str]]:
    raw = str(folder_ref or "").strip()
    if not raw:
        return [("", 0, None, "(empty folder reference)")]

    paths = None
    for rel_key in reference_lookup_keys(raw):
        paths = clip_index.by_dir_rel_path.get(rel_key)
        if paths is not None:
            break
    if paths is None:
        matches = clip_index.by_dir_name.get(Path(rel_key).name, [])
        if len(matches) == 1:
            paths = matches[0]
        elif len(matches) > 1:
            return [("", 0, None, f"{raw} (matched multiple folders; use relative path)")]

    if not paths:
        return [("", 0, None, raw)]

    resolved = []
    for path in paths:
        parsed = parse_clip_file_name(path.name)
        if parsed is None:
            continue
        clip_set, offset = parsed
        resolved.append((clip_set, offset, path, path.name))
    return resolved or [("", 0, None, f"{raw} (no double_*.mp4 clips found)")]


def iter_row_sources(
    row: RangeRow | SequenceRow,
    clip_index: ClipCatalog,
):
    if isinstance(row, RangeRow):
        step = 1 if row.stop >= row.start else -1
        for offset in range(row.start, row.stop + step, step):
            clip_path = clip_index.by_key.get((row.clip_set, offset))
            yield row.clip_set, offset, clip_path, f"double_{row.clip_set}_{offset:03d}.mp4", False
        return

    for entry in row.entries:
        if entry.kind == "range":
            step = 1 if entry.stop >= entry.start else -1
            for offset in range(entry.start, entry.stop + step, step):
                clip_path = clip_index.by_key.get((entry.clip_set, offset))
                yield entry.clip_set, offset, clip_path, f"double_{entry.clip_set}_{offset:03d}.mp4", True
            continue
        if entry.kind == "file":
            file_clip_set, file_offset, file_path, file_display = resolve_file_reference(entry.file_ref, clip_index)
            if file_path is not None:
                yield file_clip_set, file_offset, file_path, file_display, False
                continue
            for resolved in resolve_folder_reference(entry.file_ref, clip_index):
                clip_set, offset, clip_path, display_name = resolved
                yield clip_set, offset, clip_path, display_name, False


def load_range_rows(path: Path) -> list[RangeRow]:
    if not path.exists():
        raise RuntimeError(f"Ranges file does not exist: {path}")
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise RuntimeError(f"Ranges file is empty: {path}")

    reader = csv.DictReader(text.splitlines(), delimiter=detect_delimiter(text))
    required = {"row_number", "clip_set", "start", "stop"}
    if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
        raise RuntimeError("Ranges file must contain columns: row_number, clip_set, start, stop")

    rows: list[RangeRow] = []
    for row in reader:
        clip_set = (row.get("clip_set") or "").strip()
        if not clip_set:
            continue
        rows.append(
            RangeRow(
                row_number=int(row["row_number"]),
                clip_set=clip_set,
                start=int(row["start"]),
                stop=int(row["stop"]),
            )
        )

    if not rows:
        raise RuntimeError("No usable rows found in ranges file.")
    return rows


def build_clip_catalog_from_paths(source_root: Path, paths, label: str = "provided paths") -> ClipCatalog:
    source_root = source_root.resolve()
    log(f"Building clip index from {label}")
    index: dict[tuple[str, int], Path] = {}
    by_rel_path: dict[str, Path] = {}
    by_name: dict[str, list[Path]] = {}
    by_dir_rel_path: dict[str, list[Path]] = {}
    by_dir_name: dict[str, list[Path]] = {}
    scanned = 0

    unique_paths = sorted({Path(path).expanduser().resolve() for path in paths})
    for path in unique_paths:
        if path.suffix.lower() != ".mp4":
            continue
        try:
            rel_path_obj = path.relative_to(source_root)
        except ValueError:
            continue

        scanned += 1
        rel_path = rel_path_obj.as_posix()
        by_rel_path[rel_path] = path
        by_name.setdefault(path.name, []).append(path)
        rel_dir = path.parent.relative_to(source_root).as_posix()
        by_dir_rel_path.setdefault(rel_dir, []).append(path)
        by_dir_name.setdefault(path.parent.name, []).append(path)
        if not path.stem.startswith("double_"):
            continue
        parts = path.stem.split("_")
        if len(parts) < 4:
            continue
        clip_set = f"{parts[1]}_{parts[2]}"
        try:
            offset = int(parts[3])
        except ValueError:
            continue
        index[(clip_set, offset)] = path
        if scanned % 200 == 0:
            log(f"Scanned {scanned} files, indexed {len(index)} clips")
    log(f"Finished scan: scanned {scanned} mp4 files, indexed {len(index)} clips")
    return ClipCatalog(
        by_key=index,
        by_rel_path=by_rel_path,
        by_name=by_name,
        by_dir_rel_path=by_dir_rel_path,
        by_dir_name=by_dir_name,
    )


def build_clip_index(source_root: Path) -> ClipCatalog:
    log(f"Scanning source root: {source_root}")
    return build_clip_catalog_from_paths(source_root, source_root.rglob("*.mp4"), "source root scan")


def expand_rows(rows: list[RangeRow | SequenceRow], source_root: Path, clip_index: ClipCatalog) -> list[RowJob]:
    jobs, missing, skipped_missing = collect_jobs_and_missing(rows, source_root, clip_index)
    if missing:
        preview = "\n".join(missing[:30])
        more = f"\n... and {len(missing) - 30} more missing clips" if len(missing) > 30 else ""
        raise RuntimeError(f"Missing clips referenced by ranges file:\n{preview}{more}")
    if not jobs:
        skipped_note = ""
        if skipped_missing:
            skipped_note = "\nSkipped missing clips:\n" + "\n".join(skipped_missing[:30])
        raise RuntimeError(f"No clips were selected from the provided ranges.{skipped_note}")
    return jobs


def collect_jobs_and_missing(
    rows: list[RangeRow | SequenceRow],
    source_root: Path,
    clip_index: ClipCatalog,
) -> tuple[list[RowJob], list[str], list[str]]:
    jobs: list[RowJob] = []
    missing: list[str] = []
    skipped_missing: list[str] = []

    for row in rows:
        items: list[ClipItem] = []
        row_skipped_missing: list[str] = []
        for clip_set, offset, clip_path, display_name, optional_missing in iter_row_sources(row, clip_index):
            if clip_path is None:
                line = f"row {row.row_number:03d}: {display_name}"
                if optional_missing:
                    row_skipped_missing.append(line)
                    skipped_missing.append(line)
                else:
                    missing.append(line)
                continue
            items.append(
                ClipItem(
                    row_index=row.row_number,
                    clip_set=clip_set,
                    offset=offset,
                    path=clip_path,
                    relative_path=clip_path.relative_to(source_root).as_posix(),
                    clip_start=parse_clip_set_datetime(clip_set) + timedelta(minutes=offset),
                )
            )
        if items:
            jobs.append(RowJob(row_index=row.row_number, row=row, items=items, skipped_missing=row_skipped_missing))
    return jobs, missing, skipped_missing


def ensure_ffmpeg_available() -> str:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg was not found in PATH.")
    return ffmpeg


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


def probe_export_profile(path: Path) -> tuple[int, int, str]:
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return 1920, 1080, "30"
    result = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,r_frame_rate",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return 1920, 1080, "30"
    try:
        data = json.loads(result.stdout)
        stream = data["streams"][0]
        return int(stream.get("width", 1920)), int(stream.get("height", 1080)), str(stream.get("r_frame_rate", "30/1"))
    except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError):
        return 1920, 1080, "30"


def format_seconds(total_seconds: float) -> str:
    rounded = int(total_seconds)
    hours, remainder = divmod(rounded, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def slugify_output_name(value: str) -> str:
    cleaned = "".join(ch.lower() if ch.isalnum() else "_" for ch in str(value or "").strip())
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned.strip("_")


def row_folder_name(job: RowJob) -> str:
    if isinstance(job.row, RangeRow):
        return f"row_{job.row_index:03d}_{job.row.clip_set}_{job.row.start:03d}_{job.row.stop:03d}"

    custom_name = slugify_output_name(job.row.output_name)
    if custom_name:
        return f"row_{job.row_index:03d}_{custom_name}"

    first = job.items[0]
    last = job.items[-1]
    if len(job.items) == 1:
        return f"row_{job.row_index:03d}_{first.clip_set}_{first.offset:03d}"
    return (
        f"row_{job.row_index:03d}_{first.clip_set}_{first.offset:03d}"
        f"_to_{last.clip_set}_{last.offset:03d}_{len(job.items):03d}clips"
    )


def resolve_output_root_for_job(job: RowJob, output_root: Path | None) -> Path:
    if output_root is not None:
        return output_root

    clip_path = job.items[0].path
    for parent in [clip_path.parent, *clip_path.parents]:
        if parent.name in {"step", "flip"}:
            return parent.parent / "event"
    return clip_path.parent / "event"


def build_output_paths(output_root: Path | None, job: RowJob) -> tuple[Path, Path, Path, Path]:
    resolved_output_root = resolve_output_root_for_job(job, output_root)
    folder_name = row_folder_name(job)
    output_dir = resolved_output_root / folder_name
    return output_dir, output_dir / f"{folder_name}.mp4", output_dir / "note.txt", output_dir / "manifest.json"


def build_note(job: RowJob, durations: list[float], output_video_name: str) -> str:
    if isinstance(job.row, RangeRow):
        request_lines = [
            f"mode: range",
            f"clip_set: {job.row.clip_set}",
            f"start: {job.row.start}",
            f"stop: {job.row.stop}",
        ]
    else:
        request_lines = [f"mode: sequence"]
        if job.row.output_name:
            request_lines.append(f"output_name: {job.row.output_name}")
        request_lines.append(f"entries: {len(job.row.entries)}")
        for entry_index, entry in enumerate(job.row.entries, start=1):
            if entry.kind == "range":
                request_lines.append(f"entry_{entry_index:02d}: range {entry.clip_set} {entry.start}->{entry.stop}")
            else:
                request_lines.append(f"entry_{entry_index:02d}: file {entry.file_ref}")
    lines = [
        f"output_video: {output_video_name}",
        f"row_index: {job.row_index}",
        *request_lines,
        f"segments: {len(job.items)}",
        f"skipped_missing: {len(job.skipped_missing)}",
        "",
        "timeline:",
    ]
    cursor = 0.0
    for index, (item, duration) in enumerate(zip(job.items, durations), start=1):
        lines.append(
            " | ".join(
                [
                    f"{index:03d}",
                    f"output {format_seconds(cursor)} -> {format_seconds(cursor + duration)}",
                    f"source_time {item.clip_start.isoformat(sep=' ')}",
                    f"file {item.relative_path}",
                ]
            )
        )
        cursor += duration
    if job.skipped_missing:
        lines.extend(["", "skipped_missing:"])
        lines.extend(job.skipped_missing)
    return "\n".join(lines) + "\n"


def build_manifest(job: RowJob, durations: list[float], output_video: Path) -> dict:
    cursor = 0.0
    segments = []
    for index, (item, duration) in enumerate(zip(job.items, durations), start=1):
        segments.append(
            {
                "index": index,
                "row_index": job.row_index,
                "clip_set": item.clip_set,
                "offset": item.offset,
                "source_file": item.relative_path,
                "source_time": item.clip_start.isoformat(sep=" "),
                "duration_seconds": round(duration, 3),
                "output_start_seconds": round(cursor, 3),
                "output_end_seconds": round(cursor + duration, 3),
            }
        )
        cursor += duration
    manifest = {
        "output_video": output_video.name,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "row_index": job.row_index,
        "segments": segments,
        "skipped_missing": job.skipped_missing,
    }
    if isinstance(job.row, RangeRow):
        manifest.update(
            {
                "mode": "range",
                "clip_set": job.row.clip_set,
                "start": job.row.start,
                "stop": job.row.stop,
            }
        )
    else:
        manifest.update(
            {
                "mode": "sequence",
                "output_name": job.row.output_name,
                "entries": [
                    {
                        "kind": entry.kind,
                        "clip_set": entry.clip_set,
                        "start": entry.start,
                        "stop": entry.stop,
                        "file_ref": entry.file_ref,
                    }
                    for entry in job.row.entries
                ],
            }
        )
    return manifest


def render_segments(
    items: list[ClipItem],
    output_dir: Path,
    ffmpeg: str,
    width: int,
    height: int,
    fps: str,
    crf: int,
    preset: str,
    progress_callback=None,
) -> list[Path]:
    segments_dir = output_dir / "segments"
    segments_dir.mkdir(exist_ok=True)
    rendered: list[Path] = []
    for index, item in enumerate(items, start=1):
        segment_path = segments_dir / f"segment_{index:04d}.mp4"
        log(f"Rendering {index}/{len(items)}: {item.relative_path}")
        if progress_callback is not None:
            progress_callback(
                {
                    "phase": "rendering",
                    "current_clip": index,
                    "total_clips": len(items),
                    "current_file": item.relative_path,
                }
            )
        result = subprocess.run(
            [
                ffmpeg,
                "-y",
                "-i",
                str(item.path),
                "-an",
                "-vf",
                f"scale={width}:{height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,setsar=1",
                "-r",
                fps,
                "-c:v",
                "libx264",
                "-preset",
                preset,
                "-crf",
                str(crf),
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(segment_path),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg failed on {item.relative_path}\n{result.stderr.strip()}")
        rendered.append(segment_path)
    return rendered


def concat_segments(segment_paths: list[Path], output_video: Path, ffmpeg: str) -> None:
    concat_file = output_video.parent / "concat_list.txt"
    concat_file.write_text("\n".join(f"file '{path.as_posix()}'" for path in segment_paths) + "\n", encoding="utf-8")
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
        raise RuntimeError(f"ffmpeg concat failed\n{result.stderr.strip()}")


def cleanup_render_segments(output_dir: Path) -> None:
    segments_dir = output_dir / "segments"
    if segments_dir.exists():
        shutil.rmtree(segments_dir, ignore_errors=True)


def build_videos_from_rows(
    rows: list[RangeRow | SequenceRow],
    source_root: Path,
    output_root: Path | None = None,
    crf: int = 0,
    preset: str = "veryslow",
    default_clip_seconds: float = 60.0,
    dry_run: bool = False,
    progress_callback=None,
    clip_index: ClipCatalog | None = None,
    allow_missing: bool = True,
) -> dict:
    source_root = source_root.resolve()
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
        skipped_note = ""
        if skipped_missing:
            skipped_note = "\nSkipped missing clips:\n" + "\n".join(skipped_missing[:30])
        raise RuntimeError(f"No clips were selected from the provided ranges.{skipped_note}")

    results = {
        "rows": len(rows),
        "jobs": len(jobs),
        "outputs": [],
        "missing": skipped_missing,
    }
    if dry_run:
        for job in jobs:
            _, output_video, _, _ = build_output_paths(output_root, job)
            item_result = {
                "row_index": job.row_index,
                "output_video": str(output_video),
                "segments": len(job.items),
                "skipped_missing": job.skipped_missing,
            }
            if isinstance(job.row, RangeRow):
                item_result.update(
                    {
                        "mode": "range",
                        "clip_set": job.row.clip_set,
                        "start": job.row.start,
                        "stop": job.row.stop,
                    }
                )
            else:
                item_result.update(
                    {
                        "mode": "sequence",
                        "output_name": job.row.output_name,
                        "entries": len(job.row.entries),
                    }
                )
            results["outputs"].append(item_result)
        return results

    ffmpeg = ensure_ffmpeg_available()
    for job_index, job in enumerate(jobs, start=1):
        output_dir, output_video, note_path, manifest_path = build_output_paths(output_root, job)
        durations = [probe_duration_seconds(item.path, default_clip_seconds) for item in job.items]
        output_dir.mkdir(parents=True, exist_ok=True)
        note_path.write_text(build_note(job, durations, output_video.name), encoding="utf-8")
        manifest_path.write_text(
            json.dumps(build_manifest(job, durations, output_video), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        width, height, fps = probe_export_profile(job.items[0].path)
        log(f"Row {job.row_index:03d} export profile: {width}x{height} @ {fps}")
        if progress_callback is not None:
            progress_payload = {
                "phase": "row_start",
                "job_index": job_index,
                "total_jobs": len(jobs),
                "row_index": job.row_index,
                "output_dir": str(output_dir),
            }
            if isinstance(job.row, RangeRow):
                progress_payload.update(
                    {
                        "mode": "range",
                        "clip_set": job.row.clip_set,
                        "start": job.row.start,
                        "stop": job.row.stop,
                    }
                )
            else:
                progress_payload.update(
                    {
                        "mode": "sequence",
                        "output_name": job.row.output_name,
                        "entries": len(job.row.entries),
                    }
                )
            progress_callback(
                progress_payload
            )
        try:
            segment_paths = render_segments(
                job.items,
                output_dir,
                ffmpeg,
                width,
                height,
                fps,
                crf,
                preset,
                progress_callback=progress_callback,
            )
            if progress_callback is not None:
                progress_callback(
                    {
                        "phase": "concat",
                        "job_index": job_index,
                        "total_jobs": len(jobs),
                        "row_index": job.row_index,
                        "output_video": str(output_video),
                    }
                )
            concat_segments(segment_paths, output_video, ffmpeg)
        finally:
            cleanup_render_segments(output_dir)
        item_result = {
            "row_index": job.row_index,
            "output_video": str(output_video),
            "note_path": str(note_path),
            "manifest_path": str(manifest_path),
            "segments": len(job.items),
            "skipped_missing": job.skipped_missing,
        }
        if isinstance(job.row, RangeRow):
            item_result.update(
                {
                    "mode": "range",
                    "clip_set": job.row.clip_set,
                    "start": job.row.start,
                    "stop": job.row.stop,
                }
            )
        else:
            item_result.update(
                {
                    "mode": "sequence",
                    "output_name": job.row.output_name,
                    "entries": len(job.row.entries),
                }
            )
        results["outputs"].append(item_result)
        if progress_callback is not None:
            progress_callback({"phase": "row_done", **item_result, "job_index": job_index, "total_jobs": len(jobs)})
    return results


def main() -> int:
    args = parse_args()
    try:
        rows = load_range_rows(args.ranges_file)
        source_root = args.source_root.resolve()
        jobs = expand_rows(rows, source_root, build_clip_index(source_root))
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    if args.dry_run:
        log("Dry-run mode: no files will be written")
        print(f"Rows   : {len(rows)}")
        print(f"Videos : {len(jobs)}")
        for job in jobs:
            _, output_video, _, _ = build_output_paths(args.output_root.resolve(), job)
            if isinstance(job.row, RangeRow):
                summary = f"{job.row.clip_set} {job.row.start}->{job.row.stop}"
            else:
                summary = job.row.output_name or f"sequence ({len(job.row.entries)} entries)"
            print(f"row {job.row_index:03d}: {summary} -> {output_video}")
        return 0

    try:
        results = build_videos_from_rows(
            rows=rows,
            source_root=source_root,
            output_root=args.output_root.resolve(),
            crf=args.crf,
            preset=args.preset,
            default_clip_seconds=args.default_clip_seconds,
        )
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    for item in results["outputs"]:
        print(f"Created: {item['output_video']}")
        print(f"Notes  : {item['note_path']}")
        print(f"Manifest: {item['manifest_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
