import argparse
import copy
import csv
import hashlib
import json
import mimetypes
import os
import re
import threading
import time
import uuid
from datetime import datetime, timedelta, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from build_clip_edge_frames import build_clip_index_from_paths, build_edge_frame_videos_from_rows
from build_long_video_from_clip_ranges import (
    RangeRow,
    SequenceEntry,
    SequenceRow,
    build_clip_catalog_from_paths,
    build_videos_from_rows,
)


DEFAULT_NON_FLIP = Path(os.environ.get("CAP_COM_NON_FLIP", "non-flip"))
DEFAULT_FLIP = Path(os.environ.get("CAP_COM_FLIP", "flip"))


def latest_csv_or_default(prefix, fallback_name):
    base_dir = Path(__file__).resolve().parent
    paths = [path for path in base_dir.glob(f"{prefix}*.csv") if not path.name.startswith("._")]
    if not paths:
        return base_dir / fallback_name
    return max(paths, key=lambda path: path.name)


DEFAULT_COMMAND_CSV = latest_csv_or_default("command_export_", "command_export_2026-06-03_13-44-27.csv")
DEFAULT_REMOTE_VIDEO_CSV = Path(__file__).with_name("remote_mp4_video_list.csv")
DEFAULT_EVENT_GROUP_NOTES = Path(__file__).with_name("event_group_notes.json")
VIDEO_NAME_RE = re.compile(r"^double_(\d{8})_(\d{6})_(\d+)\.mp4$", re.IGNORECASE)
CAPTURE_ID_RE = re.compile(r"(?:^|,\s*)Capture_ID:\s*([^,]+)")
VIDEO_SEQUENCE_RE = re.compile(r"^(?P<prefix>double_\d{8}_\d{6})_(?P<index>\d+)\.mp4$", re.IGNORECASE)


FUNCTION_REGISTRY = [
    {
        "id": "play_sequence",
        "label": "Play Sequence",
        "description": "Play filtered clips continuously in timestamp order.",
    },
    {
        "id": "dual_watch",
        "label": "Two-Up Watch",
        "description": "Open two selected clips stacked top to bottom in a watch-style tab.",
    },
    {
        "id": "current_tab",
        "label": "Current Tab",
        "description": "Open the currently playing clip in a new browser tab.",
    },
]

RANGE_RE = re.compile(r"^\s*(-?\d+)\s*(?:\-|:|\.\.|to)\s*(-?\d+)\s*$", re.IGNORECASE)
CLIP_SET_RE = re.compile(r"(\d{8}_\d{6})")


def parse_video_name(path):
    match = VIDEO_NAME_RE.match(path.name)
    if not match:
        return None

    date_raw, time_raw, part_raw = match.groups()
    try:
        base_timestamp = datetime.strptime(date_raw + time_raw, "%Y%m%d%H%M%S")
        part = int(part_raw)
    except ValueError:
        return None

    timestamp = base_timestamp + timedelta(minutes=part)

    return {
        "date": timestamp.strftime("%Y-%m-%d"),
        "time": timestamp.strftime("%H:%M:%S"),
        "timestamp": timestamp.isoformat(),
        "part": part,
        "sort_key": (timestamp.isoformat(), part, path.name),
    }


def stable_id(kind, rel_path):
    raw = f"{kind}:{rel_path.as_posix()}".encode("utf-8", errors="replace")
    return hashlib.sha1(raw).hexdigest()[:16]


def split_group_parts(rel_path):
    parts = rel_path.parts
    if not parts:
        return "root", "root", ""

    group = parts[0]
    if len(parts) <= 1:
        return group, "root", ""

    subgroup = parts[1]
    nested_parent = "/".join(parts[1:-1])
    return group, subgroup, nested_parent


def display_kind_for_path(kind, rel_path):
    if kind == "event":
        return "event"
    if kind == "flip" and "event" in rel_path.parts:
        return "event"
    return kind


class VideoIndex:
    def __init__(self, roots):
        self.roots = roots
        self.lock = threading.RLock()
        self.items = []
        self.by_id = {}
        self.last_refresh = 0.0
        self.refresh()

    def refresh(self):
        items = []

        for kind, root in self.roots.items():
            root = root.expanduser().resolve()
            if not root.exists():
                continue

            for path in root.rglob("*.mp4"):
                if not path.is_file():
                    continue

                rel_path = path.relative_to(root)
                parsed = parse_video_name(path)
                long_timeline = load_long_timeline(path, root)
                if parsed is None and long_timeline:
                    first_timestamp = manifest_timestamp_from_segment(long_timeline[0])
                    if first_timestamp is not None:
                        parsed = {
                            "date": first_timestamp.strftime("%Y-%m-%d"),
                            "time": first_timestamp.strftime("%H:%M:%S"),
                            "timestamp": first_timestamp.isoformat(),
                            "part": 0,
                            "sort_key": (first_timestamp.isoformat(), 0, path.name),
                        }
                group, subgroup, nested_parent = split_group_parts(rel_path)
                display_kind = display_kind_for_path(kind, rel_path)
                vid = stable_id(display_kind, rel_path)

                item = {
                    "id": vid,
                    "kind": display_kind,
                    "root_kind": kind,
                    "group": group,
                    "subgroup": subgroup,
                    "nested_parent": nested_parent,
                    "name": path.name,
                    "rel_path": rel_path.as_posix(),
                    "path": str(path),
                    "url": f"/media/{vid}",
                    "size": path.stat().st_size,
                    "date": parsed["date"] if parsed else "",
                    "time": parsed["time"] if parsed else "",
                    "timestamp": parsed["timestamp"] if parsed else "",
                    "part": parsed["part"] if parsed else 0,
                    "parsed": bool(parsed),
                    "long_timeline": long_timeline,
                    "_sort_key": parsed["sort_key"] if parsed else ("9999", 0, path.name),
                }
                items.append(item)

        items.sort(key=lambda item: (item["kind"] == "event", item["group"], item["_sort_key"], item["kind"]))
        for item in items:
            item.pop("_sort_key", None)

        with self.lock:
            self.items = items
            self.by_id = {item["id"]: item for item in items}
            self.last_refresh = time.time()

    def refresh_if_stale(self, min_interval_seconds=0):
        if min_interval_seconds <= 0:
            return False

        now = time.time()
        with self.lock:
            if now - self.last_refresh < min_interval_seconds:
                return False

        self.refresh()
        return True

    def get_item(self, video_id):
        with self.lock:
            return self.by_id.get(video_id)

    def selection(self, ids):
        with self.lock:
            return [self.by_id[video_id] for video_id in ids if video_id in self.by_id]

    def filtered(self, params):
        kind = first(params, "kind", "all")
        group = first(params, "group", "all")
        subgroup = first(params, "subgroup", "all")
        date = first(params, "date", "all")
        q = first(params, "q", "").strip().lower()

        with self.lock:
            items = list(self.items)
        if kind != "all":
            items = [item for item in items if item["kind"] == kind]
        if group != "all":
            items = [item for item in items if item["group"] == group]
        if subgroup != "all":
            items = [item for item in items if item["subgroup"] == subgroup]
        if date != "all":
            items = [item for item in items if item["date"] == date]
        if q:
            items = [
                item for item in items
                if (
                    q in item["name"].lower()
                    or q in item["group"].lower()
                    or q in item["subgroup"].lower()
                    or q in item["nested_parent"].lower()
                    or q in item["rel_path"].lower()
                )
            ]

        return items

    def summary(self):
        with self.lock:
            items = list(self.items)
            last_refresh = self.last_refresh

        groups = sorted({item["group"] for item in items})
        dates = sorted({item["date"] for item in items if item["date"]})
        counts = {}
        subgroups_by_group = {}
        by_kind = {}
        for item in items:
            item_kind = item["kind"]
            counts[item_kind] = counts.get(item_kind, 0) + 1
            key = item["group"]
            subgroups_by_group.setdefault(key, set()).add(item["subgroup"])
            kind_summary = by_kind.setdefault(
                item_kind,
                {
                    "groups": set(),
                    "dates": set(),
                    "subgroups_by_group": {},
                },
            )
            kind_summary["groups"].add(item["group"])
            if item["date"]:
                kind_summary["dates"].add(item["date"])
            kind_summary["subgroups_by_group"].setdefault(item["group"], set()).add(item["subgroup"])

        return {
            "total": len(items),
            "counts": counts,
            "groups": groups,
            "subgroups_by_group": {
                group: sorted(values)
                for group, values in sorted(subgroups_by_group.items())
            },
            "by_kind": {
                kind: {
                    "groups": sorted(data["groups"]),
                    "dates": sorted(data["dates"]),
                    "subgroups_by_group": {
                        group: sorted(values)
                        for group, values in sorted(data["subgroups_by_group"].items())
                    },
                }
                for kind, data in sorted(by_kind.items())
            },
            "dates": dates,
            "functions": FUNCTION_REGISTRY,
            "last_refresh": datetime.fromtimestamp(last_refresh).isoformat() if last_refresh else "",
        }


def detect_delimiter(text):
    first_line = next((line for line in text.splitlines() if line.strip()), "")
    if "\t" in first_line:
        return "\t"
    if "|" in first_line:
        return "|"
    return ","


def iter_delimited_rows(path):
    path = Path(path)
    if not path.exists():
        return
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        first_line = handle.readline()
        delimiter = detect_delimiter(first_line)
        handle.seek(0)
        yield from csv.DictReader(handle, delimiter=delimiter)


def read_delimited_rows(path):
    return list(iter_delimited_rows(path))


def parse_utc_timestamp(value):
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(timezone.utc).replace(tzinfo=None)
    return parsed


def parse_command_arguments(value):
    parsed = {}
    for part in str(value or "").split(","):
        if ":" not in part:
            continue
        key, raw_value = part.split(":", 1)
        key = key.strip()
        if key:
            parsed[key] = raw_value.strip()
    return parsed


def parse_capture_id(arguments):
    match = CAPTURE_ID_RE.search(str(arguments or ""))
    if not match:
        return ""
    return match.group(1).strip()


def safe_float(value, default=None):
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return default


def safe_int(value, default=0):
    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return default


def command_type_from_name(value):
    return str(value or "").strip().rsplit("/", 1)[-1]


def infer_split_duration_seconds(value):
    minutes = safe_float(value, None)
    if minutes is None or minutes <= 0:
        return 60.0
    return minutes * 60.0


def pair_command_duration_seconds(events, start_type, stop_type):
    total = 0.0
    active_start = None
    for event in events:
        event_type = event.get("command_type")
        timestamp = parse_utc_timestamp(event.get("time"))
        if timestamp is None:
            continue
        if event_type == start_type:
            active_start = timestamp
        elif event_type == stop_type and active_start is not None:
            total += max(0.0, (timestamp - active_start).total_seconds())
            active_start = None
    return total


def summarize_command_events(events):
    command_counts = {}
    notable_events = []
    pump_requested_seconds = 0.0
    pump_pwm_a_total = 0.0
    pump_pwm_b_total = 0.0
    pump_start_count = 0
    torch_levels = []

    for event in events:
        command_type = event["command_type"]
        command_counts[command_type] = command_counts.get(command_type, 0) + 1
        if command_type not in {"CaptureStart", "CaptureStop"}:
            notable_events.append(command_type)
        if command_type == "PumpStart":
            pump_start_count += 1
            args = parse_command_arguments(event.get("arguments", ""))
            pump_requested_seconds += safe_float(args.get("Motor_Duration"), 0.0) or 0.0
            pump_pwm_a_total += safe_float(args.get("PWM_A"), 0.0) or 0.0
            pump_pwm_b_total += safe_float(args.get("PWM_B"), 0.0) or 0.0
        elif command_type == "TorchControl":
            args = parse_command_arguments(event.get("arguments", ""))
            level = safe_float(args.get("Torch_Percentage"), None)
            if level is not None:
                torch_levels.append(level)

    return {
        "command_counts": command_counts,
        "event_types": sorted(set(notable_events)),
        "event_count": len(events),
        "pump_start_count": pump_start_count,
        "pump_stop_count": command_counts.get("PumpStop", 0),
        "pump_command_count": pump_start_count + command_counts.get("PumpStop", 0),
        "pump_requested_seconds": round(pump_requested_seconds, 3),
        "pump_observed_seconds": round(pair_command_duration_seconds(events, "PumpStart", "PumpStop"), 3),
        "pump_avg_pwm_a": round(pump_pwm_a_total / pump_start_count, 2) if pump_start_count else 0,
        "pump_avg_pwm_b": round(pump_pwm_b_total / pump_start_count, 2) if pump_start_count else 0,
        "torch_command_count": command_counts.get("TorchControl", 0),
        "torch_min": min(torch_levels) if torch_levels else None,
        "torch_max": max(torch_levels) if torch_levels else None,
        "torch_avg": round(sum(torch_levels) / len(torch_levels), 2) if torch_levels else None,
    }


def csv_kind(path):
    name = path.name
    if name.startswith("command_export_"):
        return "command"
    if "remote_mp4_video_list" in name:
        return "remote_video"
    if name.endswith("_export_2026-06-05_03-46-20.csv") or re.match(r"^[A-Za-z_]+_export_\d{4}-", name):
        return "telemetry"
    if path.parent.name == "flip_missing_source_report" or "flip_missing_source_report" in path.parts:
        return "flip_report"
    if name == "to_long.csv":
        return "long_ranges"
    return "csv"


def relative_to_base(path, base_dir):
    try:
        return path.relative_to(base_dir).as_posix()
    except ValueError:
        return path.name


def remote_part_from_path(remote_path, directory):
    match = re.search(r"/mnt/ssd/record/([^/]+)/", str(remote_path or ""))
    if match:
        return match.group(1)
    return Path(str(directory or "")).name


def parse_remote_video_timestamp(file_name):
    match = VIDEO_NAME_RE.match(str(file_name or ""))
    if not match:
        return None, None
    date_raw, time_raw, part_raw = match.groups()
    try:
        base = datetime.strptime(date_raw + time_raw, "%Y%m%d%H%M%S")
        part = int(part_raw)
    except ValueError:
        return None, None
    return base + timedelta(minutes=part), part


def parse_video_sequence(file_name):
    match = VIDEO_SEQUENCE_RE.match(str(file_name or ""))
    if not match:
        return None
    try:
        return match.group("prefix").lower(), int(match.group("index"))
    except ValueError:
        return None


def event_session_id(session):
    raw = f"{session['capture_id']}:{session['start_time']}:{session.get('sequence_number', '')}"
    return hashlib.sha1(raw.encode("utf-8", errors="replace")).hexdigest()[:16]


class EventGroupNotes:
    def __init__(self, path):
        self.path = Path(path) if path else None
        self.lock = threading.RLock()
        self.groups = {}
        self.load()

    def load(self):
        if not self.path or not self.path.exists():
            self.groups = {}
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            data = {}
        self.groups = data.get("groups", {}) if isinstance(data, dict) else {}

    def save(self):
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "groups": self.groups,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        }
        self.path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def get(self, session_id):
        with self.lock:
            value = self.groups.get(str(session_id), {})
            return {
                "name": str(value.get("name", "")),
                "note": str(value.get("note", "")),
                "updated_at": str(value.get("updated_at", "")),
            }

    def set(self, session_id, name, note):
        session_id = str(session_id or "").strip()
        if not session_id:
            raise RuntimeError("session_id is required.")
        with self.lock:
            self.groups[session_id] = {
                "name": str(name or "").strip(),
                "note": str(note or "").strip(),
                "updated_at": datetime.now().isoformat(timespec="seconds"),
            }
            self.save()
            return self.get(session_id)

    def snapshot(self):
        with self.lock:
            return copy.deepcopy(self.groups)


class EventCatalog:
    def __init__(self, command_csv, remote_video_csv):
        self.command_csv = Path(command_csv) if command_csv else None
        self.remote_video_csv = Path(remote_video_csv) if remote_video_csv else None
        self.base_dir = Path(__file__).resolve().parent
        self.lock = threading.RLock()
        self.commands = []
        self.remote_videos = []
        self.sessions = []
        self.analytics = {}
        self.analytics_cache_key = None
        self.analytics_cache = None
        self.last_refresh = 0.0
        self.refresh()

    def refresh(self):
        commands = self.load_commands()
        remote_videos = self.load_remote_videos()
        sessions = self.build_sessions(commands, remote_videos)
        analytics = self.load_analytics()
        with self.lock:
            self.commands = commands
            self.remote_videos = remote_videos
            self.sessions = sessions
            self.analytics = analytics
            self.last_refresh = time.time()

    def load_commands(self):
        rows = read_delimited_rows(self.command_csv) if self.command_csv else []
        commands = []
        for index, row in enumerate(rows, 1):
            command_name = (row.get("Command Name") or "").strip()
            timestamp = parse_utc_timestamp(row.get("Generation Time"))
            arguments = row.get("Arguments") or ""
            if not command_name or timestamp is None:
                continue
            command_type = command_type_from_name(command_name)
            parsed_args = parse_command_arguments(arguments)
            commands.append(
                {
                    "index": index,
                    "time": timestamp.isoformat(),
                    "timestamp": timestamp,
                    "command_name": command_name,
                    "command_type": command_type,
                    "arguments": arguments,
                    "args": parsed_args,
                    "capture_id": parse_capture_id(arguments),
                    "origin": row.get("Origin", ""),
                    "sequence_number": row.get("Sequence Number", ""),
                    "ack": row.get("Acknowledge_Sent", ""),
                    "completion": row.get("Completion", ""),
                    "return_value": row.get("Return Value", ""),
                }
            )
        commands.sort(key=lambda item: (item["timestamp"], item["index"]))
        return commands

    def load_remote_videos(self):
        rows = read_delimited_rows(self.remote_video_csv) if self.remote_video_csv else []
        videos = []
        for index, row in enumerate(rows, 1):
            file_name = (row.get("file_name") or "").strip()
            remote_path = (row.get("remote_path") or "").strip()
            timestamp, clip_index = parse_remote_video_timestamp(file_name)
            if not file_name or not remote_path:
                continue
            videos.append(
                {
                    "index": index,
                    "file_name": file_name,
                    "remote_path": remote_path,
                    "directory": row.get("directory", ""),
                    "part": remote_part_from_path(remote_path, row.get("directory", "")),
                    "size_bytes": int(float(row.get("size_bytes") or 0)),
                    "size_mb": float(row.get("size_mb") or 0),
                    "time": timestamp.isoformat() if timestamp else "",
                    "timestamp": timestamp,
                    "clip_index": clip_index if clip_index is not None else "",
                }
            )
        videos.sort(key=lambda item: (item["part"], item["timestamp"] or datetime.max, item["file_name"]))
        return videos

    def build_sessions(self, commands, remote_videos):
        sessions = []
        active_by_capture_id = {}
        for command in commands:
            command_type = command["command_type"]
            capture_id = command.get("capture_id") or ""

            if command_type == "CaptureStart" and capture_id:
                active = active_by_capture_id.get(capture_id)
                if active is not None:
                    active["end"] = command["timestamp"]
                    active["end_time"] = command["timestamp"].isoformat()
                    active["status"] = "interrupted_by_next_start"
                    active_by_capture_id.pop(capture_id, None)

                session = {
                    "capture_id": capture_id,
                    "start": command["timestamp"],
                    "start_time": command["timestamp"].isoformat(),
                    "end": None,
                    "end_time": "",
                    "status": "open",
                    "camera_a": command["args"].get("Camera_A", ""),
                    "camera_b": command["args"].get("Camera_B", ""),
                    "split_duration": command["args"].get("Camera_Split_Duration", ""),
                    "sequence_number": command.get("sequence_number", ""),
                    "events": [self.public_command(command)],
                }
                sessions.append(session)
                active_by_capture_id[capture_id] = session
                continue

            for session in list(active_by_capture_id.values()):
                session["events"].append(self.public_command(command))

            if command_type == "CaptureStop" and capture_id in active_by_capture_id:
                session = active_by_capture_id.pop(capture_id)
                session["end"] = command["timestamp"]
                session["end_time"] = command["timestamp"].isoformat()
                session["status"] = "complete"

        for session in active_by_capture_id.values():
            session["status"] = "open"

        videos_by_part = {}
        for video in remote_videos:
            videos_by_part.setdefault(video["part"], []).append(video)

        for session in sessions:
            self.attach_videos(session, videos_by_part.get(session["capture_id"], []))

        sessions = self.merge_continued_interrupted_sessions(sessions)

        for session in sessions:
            self.finalize_session(session)

        sessions.sort(key=lambda item: (item["start_time"], item["capture_id"]))
        return sessions

    @staticmethod
    def session_video_names(session):
        return [video["file_name"] for video in session.get("videos", [])]

    @classmethod
    def sessions_have_continued_video(cls, previous, current):
        if previous.get("status") != "interrupted_by_next_start":
            return False
        if previous.get("capture_id") != current.get("capture_id"):
            return False
        if previous.get("end_time") != current.get("start_time"):
            return False

        previous_names = cls.session_video_names(previous)
        current_names = cls.session_video_names(current)
        if not previous_names or not current_names:
            return False
        if set(previous_names) & set(current_names):
            return True

        previous_sequences = [parse_video_sequence(name) for name in previous_names]
        current_sequences = [parse_video_sequence(name) for name in current_names]
        previous_sequences = [seq for seq in previous_sequences if seq is not None]
        current_sequences = [seq for seq in current_sequences if seq is not None]
        if not previous_sequences or not current_sequences:
            return False

        previous_by_prefix = {}
        for prefix, index in previous_sequences:
            previous_by_prefix[prefix] = max(index, previous_by_prefix.get(prefix, -1))

        current_by_prefix = {}
        for prefix, index in current_sequences:
            current_by_prefix[prefix] = min(index, current_by_prefix.get(prefix, index))

        for prefix, previous_max in previous_by_prefix.items():
            current_min = current_by_prefix.get(prefix)
            if current_min is not None and current_min >= max(0, previous_max - 2):
                return True
        return False

    @staticmethod
    def unique_sorted_videos(videos):
        seen = set()
        unique = []
        for video in videos:
            key = (video.get("remote_path"), video.get("file_name"))
            if key in seen:
                continue
            seen.add(key)
            unique.append(video)
        unique.sort(key=lambda item: (item.get("time") or "", item.get("file_name") or ""))
        return unique

    @classmethod
    def merge_session_pair(cls, previous, current):
        previous["end"] = current.get("end")
        previous["end_time"] = current.get("end_time", "")
        previous["status"] = current.get("status", previous.get("status"))
        previous["events"].extend(current.get("events", []))
        previous["videos"] = cls.unique_sorted_videos(previous.get("videos", []) + current.get("videos", []))
        previous["merged_session_count"] = previous.get("merged_session_count", 1) + current.get("merged_session_count", 1)
        return previous

    @classmethod
    def merge_continued_interrupted_sessions(cls, sessions):
        sessions = sorted(sessions, key=lambda item: (item["start"], item["capture_id"]))
        merged = []
        last_index_by_capture_id = {}
        for session in sessions:
            capture_id = session.get("capture_id")
            previous_index = last_index_by_capture_id.get(capture_id)
            if previous_index is not None and cls.sessions_have_continued_video(merged[previous_index], session):
                cls.merge_session_pair(merged[previous_index], session)
            else:
                session["merged_session_count"] = 1
                last_index_by_capture_id[capture_id] = len(merged)
                merged.append(session)
        return merged

    @staticmethod
    def public_command(command):
        return {
            "time": command["time"],
            "command_name": command["command_name"],
            "command_type": command["command_type"],
            "arguments": command["arguments"],
            "capture_id": command.get("capture_id", ""),
            "origin": command.get("origin", ""),
            "sequence_number": command.get("sequence_number", ""),
        }

    @staticmethod
    def attach_videos(session, videos):
        start = session["start"] - timedelta(seconds=10)
        end = session["end"] + timedelta(seconds=90) if session.get("end") else None
        attached = []
        for video in videos:
            timestamp = video.get("timestamp")
            if timestamp is None:
                continue
            if timestamp < start:
                continue
            if end is not None and timestamp > end:
                continue
            attached.append(video)
        session["videos"] = [EventCatalog.public_video(video) for video in attached]

    @staticmethod
    def public_video(video):
        return {
            "file_name": video["file_name"],
            "remote_path": video["remote_path"],
            "part": video["part"],
            "size_bytes": video["size_bytes"],
            "size_mb": video["size_mb"],
            "time": video["time"],
            "clip_index": video["clip_index"],
        }

    @staticmethod
    def finalize_session(session):
        command_summary = summarize_command_events(session["events"])
        start = session["start"]
        end = session.get("end")
        videos = session.get("videos", [])
        split_clip_seconds = infer_split_duration_seconds(session.get("split_duration"))
        duration_seconds = int((end - start).total_seconds()) if end else None
        video_size_mb = sum(safe_float(video.get("size_mb"), 0.0) or 0.0 for video in videos)

        session["id"] = event_session_id(session)
        session["duration_seconds"] = duration_seconds
        session["event_types"] = command_summary["event_types"]
        session["command_counts"] = command_summary["command_counts"]
        session["event_count"] = command_summary["event_count"]
        session["pump_start_count"] = command_summary["pump_start_count"]
        session["pump_stop_count"] = command_summary["pump_stop_count"]
        session["pump_command_count"] = command_summary["pump_command_count"]
        session["pump_requested_seconds"] = command_summary["pump_requested_seconds"]
        session["pump_observed_seconds"] = command_summary["pump_observed_seconds"]
        session["pump_avg_pwm_a"] = command_summary["pump_avg_pwm_a"]
        session["pump_avg_pwm_b"] = command_summary["pump_avg_pwm_b"]
        session["torch_command_count"] = command_summary["torch_command_count"]
        session["torch_min"] = command_summary["torch_min"]
        session["torch_max"] = command_summary["torch_max"]
        session["torch_avg"] = command_summary["torch_avg"]
        session["command_density_per_minute"] = round(command_summary["event_count"] / (duration_seconds / 60.0), 2) if duration_seconds else None
        session["video_count"] = len(videos)
        session["estimated_clip_seconds"] = round(len(videos) * split_clip_seconds, 3)
        session["clip_seconds_per_video"] = round(split_clip_seconds, 3)
        session["total_remote_size_mb"] = round(video_size_mb, 3)
        session["first_video_time"] = videos[0]["time"] if videos else ""
        session["last_video_time"] = videos[-1]["time"] if videos else ""
        session["merged_session_count"] = session.get("merged_session_count", 1)
        session.pop("start", None)
        session.pop("end", None)

    def filtered(self, params):
        session_id = first(params, "session_id", "").strip()
        capture_id = first(params, "capture_id", "all")
        event_type = first(params, "event_type", "all")
        status = first(params, "status", "all")
        q = first(params, "q", "").strip().lower()

        with self.lock:
            sessions = list(self.sessions)

        if session_id:
            sessions = [session for session in sessions if session["id"] == session_id]
        if capture_id != "all":
            sessions = [session for session in sessions if session["capture_id"] == capture_id]
        if event_type != "all":
            sessions = [session for session in sessions if event_type in session["command_counts"]]
        if status != "all":
            sessions = [session for session in sessions if session["status"] == status]
        if q:
            sessions = [
                session for session in sessions
                if (
                    q in session["capture_id"].lower()
                    or any(q in event["arguments"].lower() for event in session["events"])
                    or any(q in video["file_name"].lower() for video in session["videos"])
                )
            ]
        return sessions

    @staticmethod
    def local_video_summary(item):
        return {
            "id": item["id"],
            "kind": item["kind"],
            "group": item["group"],
            "subgroup": item["subgroup"],
            "rel_path": item["rel_path"],
            "path": item["path"],
            "url": item["url"],
            "watch_url": f"/watch?id={item['id']}",
        }

    @staticmethod
    def local_by_name(video_index):
        matches = {}
        if video_index is None:
            return matches
        with video_index.lock:
            items = list(video_index.items)
        for item in items:
            matches.setdefault(item["name"], []).append(EventCatalog.local_video_summary(item))
        return matches

    @staticmethod
    def session_link(session, notes):
        note = notes.get(session["id"], {}) if isinstance(notes, dict) else {}
        return {
            "id": session["id"],
            "capture_id": session["capture_id"],
            "group_name": str(note.get("name", "")),
            "group_note": str(note.get("note", "")),
            "start_time": session["start_time"],
            "status": session["status"],
            "event_types": session["event_types"],
            "video_count": session["video_count"],
            "url": f"/events?session_id={session['id']}",
        }

    def enrich_sessions(self, sessions, video_index=None, notes=None):
        notes = notes or {}
        by_name = self.local_by_name(video_index)
        with self.lock:
            flip_parts = copy.deepcopy(self.analytics.get("flip_report", {}).get("parts", {}))
        enriched = copy.deepcopy(sessions)
        for session in enriched:
            group_note = notes.get(session["id"], {}) if isinstance(notes, dict) else {}
            session["group_name"] = str(group_note.get("name", ""))
            session["group_note"] = str(group_note.get("note", ""))
            local_count = 0
            for video in session.get("videos", []):
                matches = by_name.get(video["file_name"], [])
                video["local_available"] = bool(matches)
                video["local_match_count"] = len(matches)
                video["local_matches"] = matches[:5]
                if matches:
                    local_count += 1
            session["local_video_count"] = local_count
            session["local_missing_count"] = max(0, session.get("video_count", 0) - local_count)
            session["flip_report"] = flip_parts.get(str(session.get("capture_id", "")), {})
        return enriched

    def csv_paths(self):
        paths = []
        if self.base_dir.exists():
            paths.extend(self.base_dir.glob("*.csv"))
        report_dir = self.base_dir / "flip_missing_source_report"
        if report_dir.exists():
            paths.extend(report_dir.rglob("*.csv"))
        clean = []
        seen = set()
        for path in paths:
            if path.name.startswith("._") or not path.is_file():
                continue
            key = str(path.resolve())
            if key in seen:
                continue
            seen.add(key)
            clean.append(path)
        return sorted(clean, key=lambda item: relative_to_base(item, self.base_dir))

    def csv_cache_key(self, paths):
        key = []
        for path in paths:
            try:
                stat = path.stat()
            except OSError:
                continue
            key.append((str(path), stat.st_size, stat.st_mtime_ns))
        return tuple(key)

    def load_analytics(self):
        paths = self.csv_paths()
        cache_key = self.csv_cache_key(paths)
        if cache_key == self.analytics_cache_key and self.analytics_cache is not None:
            return copy.deepcopy(self.analytics_cache)

        analytics = {
            "csv_files": self.summarize_csv_files(paths),
            "commands": self.summarize_all_command_csvs(paths),
            "remote_videos": self.summarize_all_remote_video_csvs(paths),
            "flip_report": self.summarize_flip_report(paths),
            "telemetry": self.summarize_telemetry_csvs(paths),
            "long_ranges": self.summarize_long_ranges(paths),
        }
        analytics["csv_file_count"] = len(analytics["csv_files"])
        self.analytics_cache_key = cache_key
        self.analytics_cache = copy.deepcopy(analytics)
        return analytics

    def summarize_csv_files(self, paths):
        files = []
        for path in paths:
            try:
                stat = path.stat()
            except OSError:
                continue
            files.append({
                "name": path.name,
                "rel_path": relative_to_base(path, self.base_dir),
                "kind": csv_kind(path),
                "size_bytes": stat.st_size,
                "modified_time": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            })
        return files

    def summarize_all_command_csvs(self, paths):
        command_paths = [path for path in paths if path.name.startswith("command_export_")]
        seen = set()
        commands = []
        for path in command_paths:
            for index, row in enumerate(iter_delimited_rows(path), 1):
                command_name = (row.get("Command Name") or "").strip()
                timestamp = parse_utc_timestamp(row.get("Generation Time"))
                if not command_name or timestamp is None:
                    continue
                unique_key = (
                    row.get("Generation Time", ""),
                    command_name,
                    row.get("Arguments", ""),
                    row.get("Sequence Number", ""),
                    row.get("Binary", ""),
                )
                if unique_key in seen:
                    continue
                seen.add(unique_key)
                commands.append({
                    "index": index,
                    "timestamp": timestamp,
                    "time": timestamp.isoformat(),
                    "command_name": command_name,
                    "command_type": command_type_from_name(command_name),
                    "arguments": row.get("Arguments") or "",
                })
        commands.sort(key=lambda item: (item["timestamp"], item["index"]))
        command_summary = summarize_command_events(commands)
        command_counts = command_summary["command_counts"]
        top_commands = sorted(command_counts.items(), key=lambda item: (-item[1], item[0]))[:8]
        return {
            "csv_count": len(command_paths),
            "total": len(commands),
            "command_counts": command_counts,
            "top_commands": [{"name": name, "count": count} for name, count in top_commands],
            "pump_start_count": command_summary["pump_start_count"],
            "pump_stop_count": command_summary["pump_stop_count"],
            "pump_command_count": command_summary["pump_command_count"],
            "pump_requested_seconds": command_summary["pump_requested_seconds"],
            "pump_observed_seconds": command_summary["pump_observed_seconds"],
            "torch_command_count": command_summary["torch_command_count"],
            "first_time": commands[0]["time"] if commands else "",
            "last_time": commands[-1]["time"] if commands else "",
        }

    def summarize_all_remote_video_csvs(self, paths):
        remote_paths = [path for path in paths if "remote_mp4_video_list" in path.name]
        seen = set()
        videos = []
        part_counts = {}
        for path in remote_paths:
            for row in iter_delimited_rows(path):
                file_name = (row.get("file_name") or "").strip()
                remote_path = (row.get("remote_path") or "").strip()
                if not file_name or not remote_path:
                    continue
                unique_key = (remote_path, file_name)
                if unique_key in seen:
                    continue
                seen.add(unique_key)
                timestamp, clip_index = parse_remote_video_timestamp(file_name)
                part = remote_part_from_path(remote_path, row.get("directory", ""))
                size_bytes = safe_int(row.get("size_bytes"), 0)
                videos.append({
                    "file_name": file_name,
                    "remote_path": remote_path,
                    "part": part,
                    "size_bytes": size_bytes,
                    "size_mb": safe_float(row.get("size_mb"), 0.0) or 0.0,
                    "timestamp": timestamp,
                    "time": timestamp.isoformat() if timestamp else "",
                    "clip_index": clip_index,
                })
                part_counts[part] = part_counts.get(part, 0) + 1
        videos.sort(key=lambda item: (item["timestamp"] or datetime.max, item["file_name"]))
        total_size_bytes = sum(video["size_bytes"] for video in videos)
        top_parts = sorted(part_counts.items(), key=lambda item: (-item[1], item[0]))[:12]
        return {
            "csv_count": len(remote_paths),
            "total": len(videos),
            "total_size_bytes": total_size_bytes,
            "total_size_mb": round(total_size_bytes / 1024 / 1024, 3),
            "estimated_clip_seconds": len(videos) * 60,
            "part_count": len(part_counts),
            "top_parts": [{"part": part, "count": count} for part, count in top_parts],
            "first_time": videos[0]["time"] if videos else "",
            "last_time": videos[-1]["time"] if videos else "",
        }

    def summarize_flip_report(self, paths):
        report_dir = self.base_dir / "flip_missing_source_report"
        summary_path = report_dir / "summary.csv"
        by_source_path = report_dir / "summary_by_source.csv"
        by_part = {}
        total = {}
        if summary_path.exists():
            for row in iter_delimited_rows(summary_path):
                part = str(row.get("part") or "").strip()
                values = {
                    "part": part,
                    "expected_remote_files": safe_int(row.get("expected_remote_files"), 0),
                    "present_in_flip": safe_int(row.get("present_in_flip"), 0),
                    "missing_from_flip": safe_int(row.get("missing_from_flip"), 0),
                    "missing_found_in_sources": safe_int(row.get("missing_found_in_sources"), 0),
                    "still_missing": safe_int(row.get("still_missing"), 0),
                    "still_missing_size_bytes": safe_int(row.get("still_missing_size_bytes"), 0),
                    "still_missing_size": row.get("still_missing_size", ""),
                }
                if part.upper() == "TOTAL":
                    total = values
                elif part:
                    by_part[part] = values
        sources = []
        if by_source_path.exists():
            for row in iter_delimited_rows(by_source_path):
                sources.append({
                    "source_root": row.get("source_root", ""),
                    "found_rows": safe_int(row.get("found_rows"), 0),
                    "unique_files": safe_int(row.get("unique_files"), 0),
                    "remote_size_bytes": safe_int(row.get("remote_size_bytes"), 0),
                    "remote_size": row.get("remote_size", ""),
                    "output_csv": row.get("output_csv", ""),
                })
        sources.sort(key=lambda item: (-item["unique_files"], item["source_root"]))
        report_csvs = [path for path in paths if "flip_missing_source_report" in path.parts]
        return {
            "csv_count": len(report_csvs),
            "parts": by_part,
            "total": total,
            "source_count": len(sources),
            "top_sources": sources[:8],
        }

    def summarize_telemetry_csvs(self, paths):
        metrics = []
        for path in paths:
            if not re.match(r"^[A-Za-z_]+_export_\d{4}-", path.name):
                continue
            if path.name.startswith("command_export_"):
                continue
            rows = iter_delimited_rows(path)
            first_row = next(rows, None)
            if not first_row or "Time" not in first_row:
                continue
            metric_columns = [column for column in first_row.keys() if column != "Time"]
            if not metric_columns:
                continue
            metric_column = metric_columns[0]
            summary = self.summarize_telemetry_rows(first_row, rows, metric_column)
            summary.update({
                "name": metric_column,
                "csv": relative_to_base(path, self.base_dir),
            })
            metrics.append(summary)
        metrics.sort(key=lambda item: item["name"])
        return metrics

    def summarize_telemetry_rows(self, first_row, rows, metric_column):
        count = 0
        numeric_count = 0
        nonzero_count = 0
        total = 0.0
        min_value = None
        max_value = None
        first_time = ""
        last_time = ""
        last_value = None
        def consume(row):
            nonlocal count, numeric_count, nonzero_count, total, min_value, max_value, first_time, last_time, last_value
            count += 1
            timestamp = row.get("Time", "")
            if timestamp:
                if not first_time:
                    first_time = timestamp
                last_time = timestamp
            value = safe_float(row.get(metric_column), None)
            if value is None:
                return
            numeric_count += 1
            total += value
            if value != 0:
                nonzero_count += 1
            min_value = value if min_value is None else min(min_value, value)
            max_value = value if max_value is None else max(max_value, value)
            last_value = value

        consume(first_row)
        for row in rows:
            consume(row)
        return {
            "row_count": count,
            "numeric_count": numeric_count,
            "nonzero_count": nonzero_count,
            "min": min_value,
            "max": max_value,
            "avg": round(total / numeric_count, 3) if numeric_count else None,
            "last": last_value,
            "first_time": first_time,
            "last_time": last_time,
        }

    def summarize_long_ranges(self, paths):
        long_paths = [path for path in paths if path.name == "to_long.csv"]
        total_rows = 0
        total_clip_ranges = 0
        clip_sets = set()
        for path in long_paths:
            for row in iter_delimited_rows(path):
                total_rows += 1
                clip_set = str(row.get("clip_set") or "").strip()
                if clip_set:
                    clip_sets.add(clip_set)
                start = safe_int(row.get("start"), None)
                stop = safe_int(row.get("stop"), None)
                if start is not None and stop is not None:
                    total_clip_ranges += abs(stop - start) + 1
        return {
            "csv_count": len(long_paths),
            "row_count": total_rows,
            "unique_clip_sets": len(clip_sets),
            "estimated_source_clips": total_clip_ranges,
        }

    def related_sessions_for_videos(self, videos, notes=None):
        notes = notes or {}
        with self.lock:
            sessions = list(self.sessions)

        sessions_by_file = {}
        for session in sessions:
            public_session = self.session_link(session, notes)
            for video in session.get("videos", []):
                sessions_by_file.setdefault(video["file_name"], []).append(public_session)

        enriched = []
        for video in videos:
            item = dict(video)
            related = sessions_by_file.get(item.get("name", ""), [])
            seen = set()
            item["related_event_sessions"] = []
            for session in related:
                if session["id"] in seen:
                    continue
                seen.add(session["id"])
                item["related_event_sessions"].append(session)
            enriched.append(item)
        return enriched

    def summary(self):
        with self.lock:
            sessions = list(self.sessions)
            remote_videos = list(self.remote_videos)
            analytics = copy.deepcopy(self.analytics)
            last_refresh = self.last_refresh

        capture_ids = sorted({session["capture_id"] for session in sessions}, key=lambda x: (len(x), x))
        event_types = sorted({event["command_type"] for session in sessions for event in session["events"]})
        statuses = sorted({session["status"] for session in sessions})
        total_videos = sum(session["video_count"] for session in sessions)
        total_clip_seconds = sum(session.get("estimated_clip_seconds", 0) for session in sessions)
        total_remote_size_mb = sum(session.get("total_remote_size_mb", 0) for session in sessions)
        total_pump_requested_seconds = sum(session.get("pump_requested_seconds", 0) for session in sessions)
        total_pump_observed_seconds = sum(session.get("pump_observed_seconds", 0) for session in sessions)
        total_pump_starts = sum(session.get("pump_start_count", 0) for session in sessions)
        total_command_events = sum(session.get("event_count", 0) for session in sessions)
        return {
            "command_csv": str(self.command_csv) if self.command_csv else "",
            "remote_video_csv": str(self.remote_video_csv) if self.remote_video_csv else "",
            "session_count": len(sessions),
            "remote_video_count": len(remote_videos),
            "session_video_count": total_videos,
            "session_clip_seconds": round(total_clip_seconds, 3),
            "session_remote_size_mb": round(total_remote_size_mb, 3),
            "session_pump_requested_seconds": round(total_pump_requested_seconds, 3),
            "session_pump_observed_seconds": round(total_pump_observed_seconds, 3),
            "session_pump_start_count": total_pump_starts,
            "session_command_event_count": total_command_events,
            "analytics": analytics,
            "capture_ids": capture_ids,
            "event_types": event_types,
            "statuses": statuses,
            "last_refresh": datetime.fromtimestamp(last_refresh).isoformat() if last_refresh else "",
        }


def normalize_clip_set(value):
    raw = str(value or "").strip()
    if not raw:
        raise RuntimeError("Missing clip prefix / clip_set value.")
    match = CLIP_SET_RE.search(raw)
    if match:
        return match.group(1)
    raise RuntimeError(f"Clip prefix must contain YYYYMMDD_HHMMSS: {raw}")


def parse_range_value(value):
    raw = str(value or "").strip()
    if not raw:
        raise RuntimeError("Missing range value.")
    single = re.fullmatch(r"-?\d+", raw)
    if single:
        point = int(raw)
        return point, point
    match = RANGE_RE.match(raw.replace(" ", ""))
    if match:
        return int(match.group(1)), int(match.group(2))
    raise RuntimeError(f"Range must look like 12-34 or 12:34: {raw}")


def parse_sequence_entries(sequence_text):
    text = str(sequence_text or "").strip()
    if not text:
        raise RuntimeError("Sequence items are required.")

    entries = []
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue

        if line.lower().endswith(".mp4") or line.lower().startswith("double_"):
            entries.append(SequenceEntry(kind="file", file_ref=line))
            continue

        values = [part.strip() for part in re.split(r"[\t,|]+|\s{2,}|\s+", line) if part.strip()]
        if len(values) == 1:
            entries.append(SequenceEntry(kind="file", file_ref=line))
            continue
        if len(values) < 2:
            raise RuntimeError(
                f"Sequence line {line_number}: use `20260527_153639 0-5`, `double_20260527_153639_003.mp4`, or a flip folder path."
            )
        clip_set = normalize_clip_set(values[0])
        start, stop = parse_range_value(values[1])
        entries.append(SequenceEntry(kind="range", clip_set=clip_set, start=start, stop=stop))

    if not entries:
        raise RuntimeError("Sequence items are empty.")
    return entries


def parse_int_field(value, field_name):
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = int(text)
    except ValueError as exc:
        raise RuntimeError(f"{field_name} must be a whole number.") from exc
    if parsed < 1:
        raise RuntimeError(f"{field_name} must be 1 or greater.")
    return parsed - 1


def parse_required_positive_int(value, field_name):
    text = str(value or "").strip()
    if not text:
        raise RuntimeError(f"{field_name} is required.")
    try:
        parsed = int(text)
    except ValueError as exc:
        raise RuntimeError(f"{field_name} must be a whole number.") from exc
    if parsed < 1:
        raise RuntimeError(f"{field_name} must be 1 or greater.")
    return parsed


def parse_required_non_negative_int(value, field_name):
    text = str("" if value is None else value).strip()
    if not text:
        raise RuntimeError(f"{field_name} is required.")
    try:
        parsed = int(text)
    except ValueError as exc:
        raise RuntimeError(f"{field_name} must be a whole number.") from exc
    if parsed < 0:
        raise RuntimeError(f"{field_name} must be 0 or greater.")
    return parsed


def parse_bool_field(value, default=False):
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on", "require", "required", "strict"}:
        return True
    if text in {"0", "false", "no", "n", "off", "skip", "allow", "optional"}:
        return False
    return default


def parse_preset_field(value, field_name):
    preset = str(value or "veryslow").strip() or "veryslow"
    allowed = {"ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow", "slower", "veryslow"}
    if preset not in allowed:
        raise RuntimeError(f"{field_name} must be one of: {', '.join(sorted(allowed))}.")
    return preset


def detect_row_delimiter(text, preferred):
    if preferred in {"tab", "comma", "pipe"}:
        return {"tab": "\t", "comma": ",", "pipe": "|"}[preferred]
    first_line = next((line for line in text.splitlines() if line.strip()), "")
    if "\t" in first_line:
        return "\t"
    if "|" in first_line:
        return "|"
    if "," in first_line:
        return ","
    return None


def split_row_values(line, delimiter):
    if delimiter is None:
        return [part for part in line.strip().split() if part]
    return [part.strip() for part in line.split(delimiter)]


def parse_build_request(payload):
    rows_text = str(payload.get("rowsText") or "").strip()
    build_mode = str(payload.get("buildMode") or "single").strip().lower()
    output_name = str(payload.get("outputName") or "").strip()
    sequence_text = str(payload.get("sequenceText") or "").strip()
    row_number_value = parse_required_positive_int(payload.get("rowNumber"), "Row number") if not rows_text else None
    clip_prefix = str(payload.get("clipPrefix") or "").strip()
    range_text = str(payload.get("fileRange") or "").strip()
    row_number_column = parse_int_field(payload.get("rowNumberColumn"), "Row number column")
    prefix_column = parse_int_field(payload.get("prefixColumn"), "Prefix column")
    range_column = parse_int_field(payload.get("rangeColumn"), "Range column")
    delimiter = detect_row_delimiter(rows_text, str(payload.get("delimiter") or "auto").strip().lower())
    require_complete = parse_bool_field(payload.get("requireComplete"), False)
    crf = parse_required_non_negative_int(payload.get("crf") if payload.get("crf") not in {None, ""} else 0, "CRF")
    preset = parse_preset_field(payload.get("preset"), "Preset")

    rows = []
    if rows_text:
        if row_number_column is None:
            raise RuntimeError("Row number column is required when using pasted rows.")
        for row_index, line in enumerate(rows_text.splitlines(), start=1):
            if not line.strip():
                continue
            values = split_row_values(line, delimiter)
            try:
                row_number = parse_required_positive_int(
                    values[row_number_column] if row_number_column < len(values) else "",
                    "Row number",
                )
                row_clip_set = normalize_clip_set(
                    values[prefix_column] if prefix_column is not None and prefix_column < len(values) else clip_prefix
                )
                row_range = values[range_column] if range_column is not None and range_column < len(values) else range_text
                start, stop = parse_range_value(row_range)
                rows.append(RangeRow(row_number=row_number, clip_set=row_clip_set, start=start, stop=stop))
            except Exception as exc:
                raise RuntimeError(f"Row {row_index}: {exc}") from exc
    else:
        if build_mode == "sequence":
            rows.append(
                SequenceRow(
                    row_number=row_number_value,
                    entries=parse_sequence_entries(sequence_text),
                    output_name=output_name,
                )
            )
        else:
            if not clip_prefix or not range_text:
                raise RuntimeError("Provide row number, clip prefix, and range, or paste rows to build from.")
            clip_set = normalize_clip_set(clip_prefix)
            start, stop = parse_range_value(range_text)
            rows.append(RangeRow(row_number=row_number_value, clip_set=clip_set, start=start, stop=stop))

    if not rows:
        raise RuntimeError("No rows were provided.")

    return {
        "rows": rows,
        "request": {
            "row_number": row_number_value,
            "clip_prefix": clip_prefix,
            "file_range": range_text,
            "build_mode": build_mode,
            "output_name": output_name,
            "sequence_text": sequence_text,
            "rows_text": rows_text,
            "row_number_column": None if row_number_column is None else row_number_column + 1,
            "prefix_column": None if prefix_column is None else prefix_column + 1,
            "range_column": None if range_column is None else range_column + 1,
            "delimiter": delimiter or "whitespace",
            "require_complete": require_complete,
            "crf": crf,
            "preset": preset,
        },
        "settings": {
            "allow_missing": not require_complete,
            "crf": crf,
            "preset": preset,
        },
    }


def parse_optional_positive_float(value, field_name):
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = float(text)
    except ValueError as exc:
        raise RuntimeError(f"{field_name} must be a number.") from exc
    if parsed <= 0:
        raise RuntimeError(f"{field_name} must be greater than 0.")
    return parsed


def parse_edge_build_request(payload):
    spec = parse_build_request(payload)
    fps = parse_required_positive_int(payload.get("fps") or 30, "FPS")
    frames_per_image = parse_optional_positive_float(payload.get("framesPerImage"), "Frames per image")
    frame_hold_seconds = parse_optional_positive_float(payload.get("frameHoldSeconds"), "Frame hold seconds")
    default_clip_seconds = parse_optional_positive_float(payload.get("defaultClipSeconds"), "Default clip seconds")
    jpg_quality = parse_required_positive_int(payload.get("jpgQuality") or 2, "JPG quality")
    crf = parse_required_non_negative_int(payload.get("crf") if payload.get("crf") not in {None, ""} else 0, "CRF")
    preset = parse_preset_field(payload.get("preset"), "Preset")
    source_mode = str(payload.get("sourceMode") or "video-fps").strip() or "video-fps"
    sample_fps = parse_optional_positive_float(payload.get("sampleFps"), "Sample FPS")
    if frames_per_image is None:
        frames_per_image = 1.0 if frame_hold_seconds is None else frame_hold_seconds * fps
    if default_clip_seconds is None:
        default_clip_seconds = 60.0
    if sample_fps is None:
        sample_fps = 1.0

    spec["settings"] = {
        "fps": fps,
        "frames_per_image": frames_per_image,
        "frame_hold_seconds": frame_hold_seconds,
        "default_clip_seconds": default_clip_seconds,
        "jpg_quality": jpg_quality,
        "crf": crf,
        "preset": preset,
        "source_mode": source_mode,
        "sample_fps": sample_fps,
        "allow_missing": spec["settings"]["allow_missing"],
    }
    spec["request"].update(
        {
            "fps": fps,
            "frames_per_image": frames_per_image,
            "frame_hold_seconds": frame_hold_seconds,
            "default_clip_seconds": default_clip_seconds,
            "jpg_quality": jpg_quality,
            "crf": crf,
            "preset": preset,
            "source_mode": source_mode,
            "sample_fps": sample_fps,
        }
    )
    return spec


class LongBuildManager:
    def __init__(self, source_root, index):
        self.source_root = source_root.expanduser().resolve()
        self.output_root = self.source_root.parent / "event"
        self.index = index
        self.lock = threading.RLock()
        self.jobs = {}
        self.job_order = []

    def list_jobs(self):
        with self.lock:
            ordered = [self.jobs[job_id] for job_id in reversed(self.job_order[-12:])]
            return [self._snapshot(job) for job in ordered]

    def create_job(self, payload):
        spec = parse_build_request(payload)
        now = datetime.now().isoformat(timespec="seconds")
        job_id = f"job_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        job = {
            "id": job_id,
            "status": "queued",
            "created_at": now,
            "updated_at": now,
            "completed_at": "",
            "source_root": str(self.source_root),
            "rows_total": len(spec["rows"]),
            "request": spec["request"],
            "logs": ["Queued build request."],
            "outputs": [],
            "missing": [],
            "error": "",
            "progress": {
                "phase": "queued",
                "job_index": 0,
                "total_jobs": len(spec["rows"]),
                "row_index": 0,
                "current_clip": 0,
                "total_clips": 0,
                "current_file": "",
                "message": "Queued build request.",
            },
        }
        with self.lock:
            self.jobs[job_id] = job
            self.job_order.append(job_id)
        thread = threading.Thread(target=self._run_job, args=(job_id, spec["rows"], spec["settings"]), daemon=True)
        thread.start()
        return self.get_job(job_id)

    def get_job(self, job_id):
        with self.lock:
            job = self.jobs.get(job_id)
            return None if job is None else self._snapshot(job)

    def _snapshot(self, job):
        return json.loads(json.dumps(job, ensure_ascii=False))

    def _set_job(self, job_id, **updates):
        with self.lock:
            job = self.jobs[job_id]
            job.update(updates)
            job["updated_at"] = datetime.now().isoformat(timespec="seconds")

    def _append_log(self, job_id, message):
        with self.lock:
            job = self.jobs[job_id]
            job["logs"].append(message)
            job["logs"] = job["logs"][-120:]
            job["updated_at"] = datetime.now().isoformat(timespec="seconds")

    def _parse_missing_from_error(self, message):
        missing = []
        for line in str(message).splitlines():
            text = line.strip()
            if text.startswith("row "):
                missing.append(text)
        return missing

    def _progress_message(self, update):
        phase = update.get("phase")
        if phase == "row_start":
            if update.get("mode") == "sequence":
                label = update.get("output_name") or f"{update.get('entries', 0)} entries"
                return (
                    f"Starting row {update.get('row_index', 0):03d} "
                    f"({update.get('job_index', 0)}/{update.get('total_jobs', 0)}) "
                    f"{label}"
                ).strip()
            return (
                f"Starting row {update.get('row_index', 0):03d} "
                f"({update.get('job_index', 0)}/{update.get('total_jobs', 0)}) "
                f"{update.get('clip_set', '')} {update.get('start', '')}->{update.get('stop', '')}"
            ).strip()
        if phase == "rendering":
            return f"Rendering {update.get('current_clip', 0)}/{update.get('total_clips', 0)} {update.get('current_file', '')}".strip()
        if phase == "concat":
            return f"Concatenating row {update.get('row_index', 0):03d}"
        if phase == "row_done":
            return f"Created row {update.get('row_index', 0):03d}: {Path(update.get('output_video', '')).name}"
        return str(update.get("message") or phase or "Working...")

    def _source_clip_paths(self):
        with self.index.lock:
            return [
                item["path"]
                for item in self.index.items
                if item.get("kind") == "flip" and item.get("path")
            ]

    def _run_job(self, job_id, rows, settings):
        self._set_job(job_id, status="running")

        def progress_callback(update):
            message = self._progress_message(update)
            with self.lock:
                job = self.jobs[job_id]
                job["progress"] = {
                    "phase": update.get("phase", "running"),
                    "job_index": update.get("job_index", job["progress"].get("job_index", 0)),
                    "total_jobs": update.get("total_jobs", job["rows_total"]),
                    "row_index": update.get("row_index", job["progress"].get("row_index", 0)),
                    "current_clip": update.get("current_clip", job["progress"].get("current_clip", 0)),
                    "total_clips": update.get("total_clips", job["progress"].get("total_clips", 0)),
                    "current_file": update.get("current_file", ""),
                    "message": message,
                }
                job["updated_at"] = datetime.now().isoformat(timespec="seconds")
            self._append_log(job_id, message)

        try:
            source_clip_paths = self._source_clip_paths()
            self._append_log(job_id, f"Using indexed clips under {self.source_root} ({len(source_clip_paths)} paths)")
            clip_index = build_clip_catalog_from_paths(self.source_root, source_clip_paths, "video review index")
            results = build_videos_from_rows(
                rows=rows,
                source_root=self.source_root,
                output_root=self.output_root,
                crf=settings["crf"],
                preset=settings["preset"],
                progress_callback=progress_callback,
                clip_index=clip_index,
                allow_missing=settings["allow_missing"],
            )
            self.index.refresh()
            skipped_missing = results.get("missing", [])
            done_message = f"Created {len(results['outputs'])} long file(s)."
            if skipped_missing:
                done_message += f" Skipped {len(skipped_missing)} missing clip(s)."
            self._set_job(
                job_id,
                status="completed",
                completed_at=datetime.now().isoformat(timespec="seconds"),
                outputs=results["outputs"],
                missing=skipped_missing,
                progress={
                    "phase": "done",
                    "job_index": results["jobs"],
                    "total_jobs": results["jobs"],
                    "row_index": results["jobs"],
                    "current_clip": 0,
                    "total_clips": 0,
                    "current_file": "",
                    "message": done_message,
                },
            )
            if skipped_missing:
                self._append_log(job_id, f"Skipped {len(skipped_missing)} missing clip(s) from sequence range(s).")
            self._append_log(job_id, f"Finished successfully with {len(results['outputs'])} output(s).")
        except Exception as exc:
            message = str(exc)
            missing = self._parse_missing_from_error(message)
            self._set_job(
                job_id,
                status="failed",
                completed_at=datetime.now().isoformat(timespec="seconds"),
                error=message,
                missing=missing,
                progress={
                    "phase": "failed",
                    "job_index": 0,
                    "total_jobs": len(rows),
                    "row_index": 0,
                    "current_clip": 0,
                    "total_clips": 0,
                    "current_file": "",
                    "message": message,
                },
            )
            self._append_log(job_id, f"Build failed: {message}")


class EdgeFrameBuildManager:
    def __init__(self, source_root, index):
        self.source_root = source_root.expanduser().resolve()
        self.output_root = self.source_root.parent / "event"
        self.index = index
        self.lock = threading.RLock()
        self.jobs = {}
        self.job_order = []

    def list_jobs(self):
        with self.lock:
            ordered = [self.jobs[job_id] for job_id in reversed(self.job_order[-12:])]
            return [self._snapshot(job) for job in ordered]

    def create_job(self, payload):
        spec = parse_edge_build_request(payload)
        now = datetime.now().isoformat(timespec="seconds")
        job_id = f"edge_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        job = {
            "id": job_id,
            "status": "queued",
            "created_at": now,
            "updated_at": now,
            "completed_at": "",
            "source_root": str(self.source_root),
            "rows_total": len(spec["rows"]),
            "request": spec["request"],
            "logs": ["Queued edge-frame build request."],
            "outputs": [],
            "missing": [],
            "error": "",
            "progress": {
                "phase": "queued",
                "job_index": 0,
                "total_jobs": len(spec["rows"]),
                "row_index": 0,
                "current_clip": 0,
                "total_clips": 0,
                "current_file": "",
                "message": "Queued edge-frame build request.",
            },
        }
        with self.lock:
            self.jobs[job_id] = job
            self.job_order.append(job_id)
        thread = threading.Thread(
            target=self._run_job,
            args=(job_id, spec["rows"], spec["settings"]),
            daemon=True,
        )
        thread.start()
        return self.get_job(job_id)

    def get_job(self, job_id):
        with self.lock:
            job = self.jobs.get(job_id)
            return None if job is None else self._snapshot(job)

    def _snapshot(self, job):
        return json.loads(json.dumps(job, ensure_ascii=False))

    def _set_job(self, job_id, **updates):
        with self.lock:
            job = self.jobs[job_id]
            job.update(updates)
            job["updated_at"] = datetime.now().isoformat(timespec="seconds")

    def _append_log(self, job_id, message):
        with self.lock:
            job = self.jobs[job_id]
            job["logs"].append(message)
            job["logs"] = job["logs"][-120:]
            job["updated_at"] = datetime.now().isoformat(timespec="seconds")

    def _parse_missing_from_error(self, message):
        missing = []
        for line in str(message).splitlines():
            text = line.strip()
            if text.startswith("row "):
                missing.append(text)
        return missing

    def _progress_message(self, update):
        phase = update.get("phase")
        if phase == "row_start":
            return (
                f"Starting edge row {update.get('row_index', 0):03d} "
                f"({update.get('job_index', 0)}/{update.get('total_jobs', 0)}) "
                f"{update.get('clip_set', '')} {update.get('start', '')}->{update.get('stop', '')}"
            ).strip()
        if phase == "rendering":
            return (
                f"Capturing {update.get('current_clip', 0)}/{update.get('total_clips', 0)} "
                f"{update.get('current_file', '')}"
            ).strip()
        if phase == "concat":
            return f"Building slideshow for row {update.get('row_index', 0):03d}".strip()
        if phase == "row_done":
            return f"Created edge row {update.get('row_index', 0):03d}: {Path(update.get('output_video', '')).name}"
        return str(update.get("message") or phase or "Working...")

    def _source_clip_paths(self):
        with self.index.lock:
            return [
                item["path"]
                for item in self.index.items
                if item.get("kind") == "flip" and item.get("path")
            ]

    def _run_job(self, job_id, rows, settings):
        self._set_job(job_id, status="running")

        def progress_callback(update):
            message = self._progress_message(update)
            with self.lock:
                job = self.jobs[job_id]
                job["progress"] = {
                    "phase": update.get("phase", "running"),
                    "job_index": update.get("job_index", job["progress"].get("job_index", 0)),
                    "total_jobs": update.get("total_jobs", job["rows_total"]),
                    "row_index": update.get("row_index", job["progress"].get("row_index", 0)),
                    "current_clip": update.get("current_clip", job["progress"].get("current_clip", 0)),
                    "total_clips": update.get("total_clips", job["progress"].get("total_clips", 0)),
                    "current_file": update.get("current_file", ""),
                    "message": message,
                }
                job["updated_at"] = datetime.now().isoformat(timespec="seconds")
            self._append_log(job_id, message)

        try:
            source_clip_paths = self._source_clip_paths()
            self._append_log(job_id, f"Using indexed clips under {self.source_root} ({len(source_clip_paths)} paths)")
            clip_index = build_clip_index_from_paths(self.source_root, source_clip_paths, "video review index")
            results = build_edge_frame_videos_from_rows(
                rows=rows,
                source_root=self.source_root,
                output_root=self.output_root,
                frame_hold_seconds=settings["frame_hold_seconds"],
                frames_per_image=settings["frames_per_image"],
                render_fps=settings["fps"],
                default_clip_seconds=settings["default_clip_seconds"],
                jpg_quality=settings["jpg_quality"],
                crf=settings["crf"],
                preset=settings["preset"],
                source_mode=settings["source_mode"],
                sample_fps=settings["sample_fps"],
                progress_callback=progress_callback,
                clip_index=clip_index,
                allow_missing=settings["allow_missing"],
            )
            self.index.refresh()
            skipped_missing = results.get("missing", [])
            done_message = f"Created {len(results['outputs'])} edge-frame file(s)."
            if skipped_missing:
                done_message += f" Skipped {len(skipped_missing)} missing clip(s)."
            self._set_job(
                job_id,
                status="completed",
                completed_at=datetime.now().isoformat(timespec="seconds"),
                outputs=results["outputs"],
                missing=skipped_missing,
                progress={
                    "phase": "done",
                    "job_index": results["jobs"],
                    "total_jobs": results["jobs"],
                    "row_index": results["jobs"],
                    "current_clip": 0,
                    "total_clips": 0,
                    "current_file": "",
                    "message": done_message,
                },
            )
            if skipped_missing:
                self._append_log(job_id, f"Skipped {len(skipped_missing)} missing clip(s) from edge range(s).")
            self._append_log(job_id, f"Finished successfully with {len(results['outputs'])} output(s).")
        except Exception as exc:
            message = str(exc)
            missing = self._parse_missing_from_error(message)
            self._set_job(
                job_id,
                status="failed",
                completed_at=datetime.now().isoformat(timespec="seconds"),
                error=message,
                missing=missing,
                progress={
                    "phase": "failed",
                    "job_index": 0,
                    "total_jobs": len(rows),
                    "row_index": 0,
                    "current_clip": 0,
                    "total_clips": 0,
                    "current_file": "",
                    "message": message,
                },
            )
            self._append_log(job_id, f"Build failed: {message}")


def first(params, key, default=""):
    values = params.get(key)
    if not values:
        return default
    return values[0]


def json_bytes(data):
    return json.dumps(data, ensure_ascii=False).encode("utf-8")


def json_script_literal(data):
    return json.dumps(data, ensure_ascii=False).replace("</", "<\\/")


def parse_manifest_timestamp(value):
    if not value:
        return None
    normalized = str(value).replace(" ", "T")
    if "." in normalized:
        head, frac = normalized.split(".", 1)
        normalized = f"{head}.{frac[:6]}"
    try:
        return datetime.fromisoformat(normalized)
    except ValueError:
        return None


def manifest_timestamp_from_segment(segment):
    for key in ["source_start", "source_time", "clip_start", "base_start"]:
        timestamp = parse_manifest_timestamp(segment.get(key))
        if timestamp is not None:
            return timestamp
    return None


def long_manifest_candidates(path, root):
    candidates = [path.with_name("manifest.json")]
    if path.name.startswith("long_") and path.suffix.lower() == ".mp4":
        key = path.stem.removeprefix("long_")
        candidates.append(root.parent / key / "manifest.json")
    deduped = []
    seen = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        deduped.append(candidate)
    return deduped


def manifest_matches_video(manifest, path):
    output_video = manifest.get("output_video")
    if output_video:
        return Path(str(output_video)).name == path.name
    return path.parent == path.with_name("manifest.json").parent


def build_timeline_segment(segment):
    try:
        output_start = float(segment["output_start_seconds"])
        output_end = float(segment["output_end_seconds"])
    except (KeyError, TypeError, ValueError):
        return None

    source_start = (
        segment.get("source_start")
        or segment.get("source_time")
        or segment.get("clip_start")
        or segment.get("base_start")
        or ""
    )
    source_end = segment.get("source_end", "")

    return {
        "index": segment.get("index"),
        "source_file": segment.get("source_file", ""),
        "source_start": source_start,
        "source_end": source_end,
        "output_start_seconds": output_start,
        "output_end_seconds": output_end,
    }


def load_long_timeline(path, root):
    manifest_path = next((candidate for candidate in long_manifest_candidates(path, root) if candidate.exists()), None)
    if manifest_path is None:
        return []

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []

    if not manifest_matches_video(manifest, path):
        return []

    timeline = []
    for segment in manifest.get("segments", []):
        timeline_segment = build_timeline_segment(segment)
        if timeline_segment is not None:
            timeline.append(timeline_segment)

    timeline.sort(key=lambda segment: segment["output_start_seconds"])
    return timeline


def safe_media_path(index, video_id):
    item = index.get_item(video_id)
    if not item:
        return None

    path = Path(item["path"]).resolve()
    root_key = item.get("root_kind") or item["kind"]
    root = index.roots[root_key].expanduser().resolve()
    try:
        path.relative_to(root)
    except ValueError:
        return None

    if not path.exists() or not path.is_file():
        return None
    return path


def build_event_export_payload(server, params):
    server.index.refresh_if_stale(server.auto_refresh_seconds)
    server.event_catalog.refresh()
    sessions = server.event_catalog.filtered(params)
    return {
        "summary": server.event_catalog.summary(),
        "sessions": server.event_catalog.enrich_sessions(
            sessions,
            video_index=server.index,
            notes=server.event_group_notes.snapshot(),
        ),
        "exported_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def event_export_html(payload):
    marker = "  <script>\n"
    export_script = (
        "  <script>\n"
        f"    window.__EVENT_EXPORT_DATA = {json_script_literal(payload)};\n"
        "  </script>\n"
    )
    return EVENT_MAP_HTML.replace(marker, export_script + marker, 1)


class ReviewHandler(BaseHTTPRequestHandler):
    server_version = "VideoReviewCenter/0.1"

    def ensure_index_current(self):
        self.server.index.refresh_if_stale(self.server.auto_refresh_seconds)

    def write_body(self, body):
        try:
            self.wfile.write(body)
            return True
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            self.close_connection = True
            return False

    def do_GET(self):
        parsed = urlparse(self.path)
        route = unquote(parsed.path)
        params = parse_qs(parsed.query)

        if route == "/":
            return self.send_html(APP_HTML)
        if route == "/events":
            return self.send_html(EVENT_MAP_HTML)
        if route in ["/events-export", "/events-export.html"]:
            return self.send_html(event_export_html(build_event_export_payload(self.server, params)))
        if route == "/long-builder":
            return self.send_html(LONG_BUILDER_HTML)
        if route == "/edge-frame-builder":
            return self.send_html(EDGE_FRAME_BUILDER_HTML)
        if route == "/api/summary":
            self.ensure_index_current()
            return self.send_json(self.server.index.summary())
        if route == "/api/event-sessions":
            return self.send_json(build_event_export_payload(self.server, params))
        if route == "/api/event-summary":
            self.server.event_catalog.refresh()
            return self.send_json(self.server.event_catalog.summary())
        if route == "/api/videos":
            self.ensure_index_current()
            videos = self.server.index.filtered(params)
            videos = self.server.event_catalog.related_sessions_for_videos(
                videos,
                notes=self.server.event_group_notes.snapshot(),
            )
            return self.send_json({"videos": videos})
        if route == "/api/selection":
            self.ensure_index_current()
            ids = [value for value in first(params, "ids", "").split(",") if value]
            videos = self.server.index.selection(ids)
            videos = self.server.event_catalog.related_sessions_for_videos(
                videos,
                notes=self.server.event_group_notes.snapshot(),
            )
            return self.send_json({"videos": videos})
        if route == "/api/refresh":
            self.server.index.refresh()
            return self.send_json(self.server.index.summary())
        if route == "/api/long-jobs":
            return self.send_json({"jobs": self.server.long_builds.list_jobs()})
        if route.startswith("/api/long-jobs/"):
            job_id = route.rsplit("/", 1)[-1]
            job = self.server.long_builds.get_job(job_id)
            if job is None:
                self.send_error(HTTPStatus.NOT_FOUND, "Job not found")
                return
            return self.send_json(job)
        if route == "/api/edge-frame-jobs":
            return self.send_json({"jobs": self.server.edge_frame_builds.list_jobs()})
        if route.startswith("/api/edge-frame-jobs/"):
            job_id = route.rsplit("/", 1)[-1]
            job = self.server.edge_frame_builds.get_job(job_id)
            if job is None:
                self.send_error(HTTPStatus.NOT_FOUND, "Job not found")
                return
            return self.send_json(job)
        if route in ["/compare", "/dual-watch"]:
            return self.send_html(DUAL_WATCH_HTML)
        if route in ["/watch", "/markers"]:
            return self.send_html(TAB_HTML)
        if route.startswith("/media/"):
            video_id = route.rsplit("/", 1)[-1]
            return self.send_media(video_id)

        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def do_POST(self):
        parsed = urlparse(self.path)
        route = unquote(parsed.path)

        if route == "/api/long-jobs":
            try:
                payload = self.read_json_body()
                job = self.server.long_builds.create_job(payload)
                return self.send_json(job, status=HTTPStatus.CREATED)
            except RuntimeError as exc:
                return self.send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
            except Exception as exc:
                return self.send_json({"error": str(exc)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)
        if route == "/api/edge-frame-jobs":
            try:
                payload = self.read_json_body()
                job = self.server.edge_frame_builds.create_job(payload)
                return self.send_json(job, status=HTTPStatus.CREATED)
            except RuntimeError as exc:
                return self.send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
            except Exception as exc:
                return self.send_json({"error": str(exc)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)
        if route == "/api/event-group-notes":
            try:
                payload = self.read_json_body()
                note = self.server.event_group_notes.set(
                    payload.get("session_id"),
                    payload.get("name", ""),
                    payload.get("note", ""),
                )
                return self.send_json({"session_id": str(payload.get("session_id", "")).strip(), "note": note})
            except RuntimeError as exc:
                return self.send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
            except Exception as exc:
                return self.send_json({"error": str(exc)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)

        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def read_json_body(self):
        length = int(self.headers.get("Content-Length", "0") or "0")
        body = self.rfile.read(length) if length > 0 else b"{}"
        try:
            return json.loads(body.decode("utf-8") or "{}")
        except json.JSONDecodeError as exc:
            raise RuntimeError("Request body must be valid JSON.") from exc

    def send_html(self, content):
        body = content.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.write_body(body)

    def send_json(self, data, status=HTTPStatus.OK):
        body = json_bytes(data)
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.write_body(body)

    def send_media(self, video_id):
        path = safe_media_path(self.server.index, video_id)
        if path is None:
            self.send_error(HTTPStatus.NOT_FOUND, "Video not found")
            return

        file_size = path.stat().st_size
        range_header = self.headers.get("Range")
        start = 0
        end = file_size - 1
        status = HTTPStatus.OK

        if range_header:
            match = re.match(r"bytes=(\d*)-(\d*)", range_header)
            if match:
                start_raw, end_raw = match.groups()
                if start_raw:
                    start = int(start_raw)
                if end_raw:
                    end = min(int(end_raw), file_size - 1)
                status = HTTPStatus.PARTIAL_CONTENT

        if start >= file_size or start > end:
            self.send_error(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
            return

        content_length = end - start + 1
        mime = mimetypes.guess_type(path.name)[0] or "video/mp4"

        self.send_response(status)
        self.send_header("Content-Type", mime)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(content_length))
        if status == HTTPStatus.PARTIAL_CONTENT:
            self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
        self.end_headers()

        with path.open("rb") as handle:
            handle.seek(start)
            remaining = content_length
            while remaining > 0:
                chunk = handle.read(min(1024 * 1024, remaining))
                if not chunk:
                    break
                if not self.write_body(chunk):
                    return
                remaining -= len(chunk)

    def log_message(self, fmt, *args):
        print("[%s] %s" % (self.log_date_time_string(), fmt % args), flush=True)


APP_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Video Review Center</title>
  <style>
    :root {
      --bg: #f4f2ec;
      --ink: #151515;
      --muted: #6d6a62;
      --line: #d7d0c2;
      --panel: #fffdf7;
      --accent: #1f6f5b;
      --accent-2: #b84b35;
      --soft: #e8e2d3;
      --shadow: 0 12px 34px rgba(41, 34, 20, 0.12);
    }

    * { box-sizing: border-box; }
    body {
      margin: 0;
      background:
        linear-gradient(135deg, rgba(31, 111, 91, 0.10), transparent 35%),
        linear-gradient(315deg, rgba(184, 75, 53, 0.10), transparent 38%),
        var(--bg);
      color: var(--ink);
      font-family: ui-sans-serif, "Avenir Next", "Segoe UI", sans-serif;
    }

    button, input, select {
      font: inherit;
    }

    button {
      border: 1px solid var(--line);
      background: var(--panel);
      color: var(--ink);
      border-radius: 6px;
      padding: 8px 10px;
      cursor: pointer;
    }

    button.primary {
      background: var(--accent);
      color: white;
      border-color: var(--accent);
    }

    button.danger {
      color: white;
      background: var(--accent-2);
      border-color: var(--accent-2);
    }

    .shell {
      display: grid;
      grid-template-columns: 340px 1fr;
      min-height: 100vh;
    }

    aside {
      border-right: 1px solid var(--line);
      background: rgba(255, 253, 247, 0.82);
      padding: 20px;
      position: sticky;
      top: 0;
      height: 100vh;
      overflow: auto;
      backdrop-filter: blur(12px);
    }

    main {
      padding: 22px;
      overflow: hidden;
    }

    h1 {
      margin: 0 0 6px;
      font-size: 26px;
      line-height: 1.1;
    }

    .muted {
      color: var(--muted);
      font-size: 13px;
    }

    .field {
      display: grid;
      gap: 6px;
      margin-top: 14px;
    }

    label {
      font-size: 12px;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0;
    }

    input, select {
      width: 100%;
      border: 1px solid var(--line);
      background: white;
      border-radius: 6px;
      padding: 9px 10px;
      min-height: 38px;
    }

    textarea {
      width: 100%;
      border: 1px solid var(--line);
      background: white;
      border-radius: 6px;
      padding: 9px 10px;
      min-height: 110px;
      resize: vertical;
      font: inherit;
    }

    .toolbar, .actions, .player-actions {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      align-items: center;
    }

    .toolbar {
      justify-content: space-between;
      margin-bottom: 16px;
    }

    .stats {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 8px;
      margin-top: 16px;
    }

    .stat {
      border: 1px solid var(--line);
      background: rgba(255,255,255,0.55);
      border-radius: 6px;
      padding: 10px;
    }

    .stat strong {
      display: block;
      font-size: 21px;
    }

    .layout {
      display: grid;
      grid-template-columns: minmax(360px, 1fr) minmax(360px, 1.1fr);
      gap: 16px;
      align-items: start;
    }

    .panel {
      background: rgba(255, 253, 247, 0.86);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: var(--shadow);
      min-width: 0;
    }

    .panel-head {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: center;
      padding: 14px 14px 0;
    }

    .panel-title {
      font-size: 15px;
      font-weight: 700;
    }

    .list {
      padding: 14px;
      display: grid;
      gap: 8px;
      max-height: calc(100vh - 130px);
      overflow: auto;
    }

    .clip {
      border: 1px solid var(--line);
      border-left: 5px solid var(--accent);
      border-radius: 8px;
      background: white;
      padding: 10px;
      display: grid;
      gap: 8px;
    }

    .clip.flip {
      border-left-color: var(--accent-2);
    }

    .clip-top {
      display: flex;
      justify-content: space-between;
      gap: 10px;
    }

    .clip-name {
      font-size: 13px;
      font-weight: 700;
      overflow-wrap: anywhere;
    }

    .tag {
      display: inline-flex;
      align-items: center;
      min-height: 22px;
      padding: 2px 7px;
      border-radius: 999px;
      background: var(--soft);
      color: var(--ink);
      font-size: 12px;
      white-space: nowrap;
    }

    .event-links {
      display: flex;
      gap: 5px;
      flex-wrap: wrap;
      margin-top: 4px;
    }

    .event-link {
      display: inline-flex;
      align-items: center;
      width: max-content;
      min-height: 24px;
      padding: 2px 8px;
      border: 1px solid var(--line);
      border-radius: 999px;
      background: var(--soft);
      color: var(--ink);
      font-size: 12px;
      text-decoration: none;
    }

    video {
      display: block;
      width: 100%;
      background: #111;
      border-radius: 6px;
      aspect-ratio: 16 / 9;
    }

    .player {
      padding: 14px;
      display: grid;
      gap: 12px;
    }

    .now {
      font-size: 13px;
      color: var(--muted);
      overflow-wrap: anywhere;
    }

    .empty {
      padding: 24px;
      color: var(--muted);
      text-align: center;
      border: 1px dashed var(--line);
      border-radius: 8px;
      background: rgba(255,255,255,0.45);
    }

    .builder {
      margin-top: 18px;
      padding: 12px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: rgba(255,255,255,0.5);
      display: grid;
      gap: 10px;
    }

    .builder-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 8px;
    }

    .job-board {
      margin-top: 16px;
      display: grid;
      gap: 10px;
    }

    .job-card {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: white;
      padding: 10px;
      display: grid;
      gap: 6px;
    }

    .job-status {
      display: inline-flex;
      align-items: center;
      width: max-content;
      padding: 3px 8px;
      border-radius: 999px;
      font-size: 12px;
      background: var(--soft);
    }

    .job-status.running {
      background: #fff1c7;
    }

    .job-status.completed {
      background: #d9f2df;
    }

    .job-status.failed {
      background: #ffd8d2;
    }

    .job-meta, .job-lines {
      display: grid;
      gap: 4px;
      font-size: 12px;
    }

    .job-lines {
      max-height: 180px;
      overflow: auto;
      border-top: 1px dashed var(--line);
      padding-top: 6px;
    }

    @media (max-width: 980px) {
      .shell { grid-template-columns: 1fr; }
      aside { position: relative; height: auto; }
      .layout { grid-template-columns: 1fr; }
      .builder-grid { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <div class="shell">
    <aside>
      <h1>Video Review Center</h1>
      <div class="muted">Timeline, sequence playback, and two-up video review from the same video index.</div>

      <div class="stats">
        <div class="stat"><strong id="totalCount">0</strong><span class="muted">clips</span></div>
        <div class="stat"><strong id="flipCount">0</strong><span class="muted">flip</span></div>
        <div class="stat"><strong id="eventCount">0</strong><span class="muted">event</span></div>
        <div class="stat"><strong id="rawCount">0</strong><span class="muted">non-flip</span></div>
      </div>

      <div class="field">
        <label for="kind">Source</label>
        <select id="kind">
          <option value="all">All</option>
          <option value="flip">Flip</option>
          <option value="event">Event</option>
        </select>
      </div>

      <div class="field">
        <label for="group">Group</label>
        <select id="group"></select>
      </div>

      <div class="field">
        <label for="subgroup">Inside Group</label>
        <select id="subgroup"></select>
      </div>

      <div class="field">
        <label for="date">Date</label>
        <select id="date"></select>
      </div>

      <div class="field">
        <label for="timeMode">Time Display</label>
        <select id="timeMode">
          <option value="gmt">GMT from file</option>
          <option value="thai">GMT+7 Thailand</option>
        </select>
      </div>

      <div class="field">
        <label for="playbackRate">Speed</label>
        <select id="playbackRate">
          <option value="0.5">0.5x</option>
          <option value="1" selected>1x</option>
          <option value="1.5">1.5x</option>
          <option value="2">2x</option>
          <option value="3">3x</option>
          <option value="4">4x</option>
          <option value="8">8x</option>
          <option value="16">16x</option>
        </select>
      </div>

      <div class="field">
        <label for="search">Search</label>
        <input id="search" placeholder="folder, file name, time">
      </div>

      <div class="field">
        <label>Functions</label>
        <div id="functionList" class="actions"></div>
      </div>

      <div class="field actions">
        <button id="refresh">Refresh Index</button>
        <button id="openEventMap">Open Event Map</button>
        <button id="openLongBuilder">Open Long Builder</button>
        <button id="openEdgeFrameBuilder">Open Edge Frame Builder</button>
      </div>
    </aside>

    <main>
      <div class="toolbar">
        <div>
          <div class="panel-title">Timeline</div>
          <div id="resultInfo" class="muted">Loading...</div>
        </div>
        <div class="actions">
          <button id="playFiltered" class="primary">Play Filtered Long</button>
          <button id="openDualWatch">Open Two-Up Watch</button>
        </div>
      </div>

      <div class="layout">
        <section class="panel">
          <div class="panel-head">
            <div class="panel-title">Clips</div>
          </div>
          <div id="clipList" class="list"></div>
        </section>

        <section class="panel">
          <div class="panel-head">
            <div>
              <div class="panel-title">Player</div>
              <div id="nowPlaying" class="now">Choose a clip or play the filtered list.</div>
            </div>
          </div>
          <div class="player">
            <video id="mainVideo" controls playsinline loop></video>
            <div class="player-actions">
              <button id="prevClip">Previous</button>
              <button id="nextClip">Next</button>
              <button id="openCurrentTab">Open Current Tab</button>
              <button id="openCurrentDualWatch">Open Current Two-Up</button>
              <button id="openMarkerTab">Open Marker Tab</button>
            </div>
          </div>
        </section>
      </div>
    </main>
  </div>

  <script>
    const state = {
      summary: null,
      videos: [],
      queue: [],
      queueIndex: -1,
      current: null,
      refreshTimer: null,
      isRefreshing: false
    };

    const $ = (id) => document.getElementById(id);
    const kindEl = $("kind");
    const groupEl = $("group");
    const subgroupEl = $("subgroup");
    const dateEl = $("date");
    const timeModeEl = $("timeMode");
    const playbackRateEl = $("playbackRate");
    const searchEl = $("search");
    const clipListEl = $("clipList");
    const mainVideo = $("mainVideo");
    const AUTO_REFRESH_MS = 5000;

    async function getJson(url) {
      const res = await fetch(url);
      if (!res.ok) throw new Error(await res.text());
      return res.json();
    }

    async function postJson(url, payload) {
      const res = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(data.error || "Request failed");
      return data;
    }

    function option(value, label) {
      const opt = document.createElement("option");
      opt.value = value;
      opt.textContent = label;
      return opt;
    }

    function activeSummaryScope() {
      if (!state.summary) return { groups: [], dates: [], subgroups_by_group: {} };
      const kind = kindEl.value || "all";
      if (kind !== "all") {
        return state.summary.by_kind?.[kind] || { groups: [], dates: [], subgroups_by_group: {} };
      }
      return {
        groups: state.summary.groups || [],
        dates: state.summary.dates || [],
        subgroups_by_group: state.summary.subgroups_by_group || {}
      };
    }

    function subgroupOptions(groupValue) {
      const scope = activeSummaryScope();
      if (groupValue === "all") {
        const merged = new Set();
        Object.values(scope.subgroups_by_group || {}).forEach(values => {
          values.forEach(value => merged.add(value));
        });
        return Array.from(merged).sort();
      }
      return scope.subgroups_by_group?.[groupValue] || [];
    }

    function syncSubgroupOptions() {
      const current = subgroupEl.value || "all";
      const values = subgroupOptions(groupEl.value);
      subgroupEl.replaceChildren(option("all", "All folders"), ...values.map(value => option(value, value)));
      subgroupEl.value = values.includes(current) ? current : "all";
    }

    function syncScopedFilters({ keepCurrent = true } = {}) {
      const scope = activeSummaryScope();
      const previousGroup = keepCurrent ? (groupEl.value || "all") : "all";
      const previousDate = keepCurrent ? (dateEl.value || "all") : "all";
      const groups = scope.groups || [];
      const dates = scope.dates || [];

      groupEl.replaceChildren(option("all", "All groups"), ...groups.map(g => option(g, g)));
      groupEl.value = groups.includes(previousGroup) ? previousGroup : "all";
      syncSubgroupOptions();
      dateEl.replaceChildren(option("all", "All dates"), ...dates.map(d => option(d, d)));
      dateEl.value = dates.includes(previousDate) ? previousDate : "all";
    }

    async function loadSummary() {
      state.summary = await getJson("/api/summary");
      $("totalCount").textContent = state.summary.total || 0;
      $("flipCount").textContent = state.summary.counts.flip || 0;
      $("eventCount").textContent = state.summary.counts.event || 0;
      $("rawCount").textContent = state.summary.counts.non_flip || 0;

      syncScopedFilters({ keepCurrent: true });

      $("functionList").replaceChildren(...state.summary.functions.map(fn => {
        const btn = document.createElement("button");
        btn.textContent = fn.label;
        btn.title = fn.description;
        btn.addEventListener("click", () => runFunction(fn.id));
        return btn;
      }));
    }

    async function loadVideos() {
      const params = new URLSearchParams({
        kind: kindEl.value,
        group: groupEl.value,
        subgroup: subgroupEl.value,
        date: dateEl.value,
        q: searchEl.value
      });
      const data = await getJson(`/api/videos?${params}`);
      state.videos = data.videos;
      state.queue = state.videos.slice();
      const currentId = state.current?.id;
      if (currentId) {
        const refreshedCurrent = state.videos.find(video => video.id === currentId);
        if (refreshedCurrent) {
          state.current = refreshedCurrent;
          $("nowPlaying").textContent = `${clipLabel(state.current)} · ${state.current.name}`;
        } else {
          state.current = null;
          $("nowPlaying").textContent = "Choose a clip or play the filtered list.";
        }
      }
      if (state.current && state.queue.length) {
        state.queueIndex = Math.max(0, state.queue.findIndex(item => item.id === state.current.id));
      } else if (!state.queue.length) {
        state.queueIndex = -1;
      }
      $("resultInfo").textContent = `${state.videos.length} clips in current filter`;
      renderClips();
    }

    async function refreshData({ force = false } = {}) {
      if (state.isRefreshing) return;
      if (!force && document.visibilityState === "hidden") return;

      state.isRefreshing = true;
      try {
        if (force) {
          await getJson("/api/refresh");
        }
        await loadSummary();
        await loadVideos();
      } finally {
        state.isRefreshing = false;
      }
    }

    function activeTimeMode() {
      return timeModeEl.value || "gmt";
    }

    function activePlaybackRate() {
      const rate = Number(playbackRateEl.value);
      return Number.isFinite(rate) && rate > 0 ? rate : 1;
    }

    function applyPlaybackRate(container = document) {
      const rate = activePlaybackRate();
      container.querySelectorAll("video").forEach(video => {
        video.playbackRate = rate;
      });
    }

    function formatTime(video, mode = activeTimeMode()) {
      if (!video.timestamp) return "unknown time";
      if (mode === "thai") {
        const date = new Date(`${video.timestamp}Z`);
        if (!Number.isNaN(date.getTime())) {
          const parts = new Intl.DateTimeFormat("en-CA", {
            timeZone: "Asia/Bangkok",
            year: "numeric",
            month: "2-digit",
            day: "2-digit",
            hour: "2-digit",
            minute: "2-digit",
            second: "2-digit",
            hour12: false
          }).formatToParts(date);
          const values = Object.fromEntries(parts.map(part => [part.type, part.value]));
          return `${values.year}-${values.month}-${values.day} ${values.hour}:${values.minute}:${values.second} GMT+7`;
        }
      }
      return `${video.date} ${video.time} GMT`;
    }

    function clipLabel(video) {
      const stamp = formatTime(video);
      const groupLabel = video.subgroup && video.subgroup !== "root"
        ? `${video.group}/${video.subgroup}`
        : video.group;
      return `${stamp} · part ${video.part} · ${groupLabel}`;
    }

    function eventLabel(event) {
      const label = event.group_name ? `${event.capture_id} · ${event.group_name}` : event.capture_id;
      const types = (event.event_types || []).length ? ` · ${(event.event_types || []).join(", ")}` : "";
      return `${label}${types}`;
    }

    function relatedEventLinks(video) {
      const events = video.related_event_sessions || [];
      if (!events.length) return "";
      return `
        <div class="event-links">
          ${events.slice(0, 4).map(event => `
            <a class="event-link" href="${escapeHtml(event.url)}" target="_blank" rel="noopener">${escapeHtml(eventLabel(event))}</a>
          `).join("")}
          ${events.length > 4 ? `<span class="muted">+${events.length - 4} more</span>` : ""}
        </div>
      `;
    }

    function renderClips() {
      if (!state.videos.length) {
        clipListEl.innerHTML = `<div class="empty">No clips match this filter.</div>`;
        return;
      }

      const nodes = state.videos.map(video => {
        const item = document.createElement("article");
        item.className = `clip ${video.kind === "flip" ? "flip" : ""}`;
        item.innerHTML = `
          <div class="clip-top">
            <div>
              <div class="clip-name">${escapeHtml(video.name)}</div>
              <div class="muted">${escapeHtml(clipLabel(video))}</div>
              ${relatedEventLinks(video)}
            </div>
            <span class="tag">${video.kind}</span>
          </div>
          <div class="actions">
            <button data-action="play">Play</button>
            <button data-action="dual-watch">Two-Up</button>
            <button data-action="tab">Tab</button>
            <button data-action="marker">Marker</button>
          </div>
        `;
        item.querySelector('[data-action="play"]').addEventListener("click", () => playVideo(video));
        item.querySelector('[data-action="dual-watch"]').addEventListener("click", () => openDualWatchTab(video));
        item.querySelector('[data-action="tab"]').addEventListener("click", () => openVideoTab(video));
        item.querySelector('[data-action="marker"]').addEventListener("click", () => openMarkerTab(video));
        return item;
      });
      clipListEl.replaceChildren(...nodes);
    }

    function playVideo(video, queue = state.videos) {
      state.current = video;
      state.queue = queue.slice();
      state.queueIndex = Math.max(0, state.queue.findIndex(item => item.id === video.id));
      mainVideo.src = video.url;
      mainVideo.playbackRate = activePlaybackRate();
      mainVideo.play().catch(() => {});
      $("nowPlaying").textContent = `${clipLabel(video)} · ${video.name}`;
    }

    function playQueue(index) {
      if (!state.queue.length) return;
      const bounded = Math.max(0, Math.min(index, state.queue.length - 1));
      playVideo(state.queue[bounded], state.queue);
    }

    function dualWatchUrl(ids) {
      const uniqueIds = Array.from(new Set(ids.filter(Boolean))).slice(0, 2);
      const params = new URLSearchParams({
        ids: uniqueIds.join(","),
        time: activeTimeMode(),
        speed: activePlaybackRate()
      });
      return `/dual-watch?${params.toString()}`;
    }

    function openDualWatchTab(seedVideo = null) {
      const ids = [];
      if (seedVideo?.id && !ids.includes(seedVideo.id)) {
        ids.push(seedVideo.id);
      }
      if (state.current?.id && !ids.includes(state.current.id)) {
        ids.push(state.current.id);
      }
      window.open(dualWatchUrl(ids), "_blank", "noopener");
    }

    function openVideoTab(video = state.current) {
      if (!video) return;
      const params = new URLSearchParams({
        id: video.id,
        time: activeTimeMode(),
        speed: activePlaybackRate()
      });
      window.open(`/watch?${params.toString()}`, "_blank", "noopener");
    }

    function openMarkerTab(video = state.current) {
      if (!video) return;
      const params = new URLSearchParams({
        id: video.id,
        time: activeTimeMode(),
        speed: activePlaybackRate()
      });
      window.open(`/markers?${params.toString()}`, "_blank", "noopener");
    }

    function syncVideos(container, method) {
      container.querySelectorAll("video").forEach(video => {
        video.playbackRate = activePlaybackRate();
        if (method === "play") video.play().catch(() => {});
        if (method === "pause") video.pause();
      });
    }

    function runFunction(id) {
      if (id === "play_sequence") {
        if (state.videos.length) playVideo(state.videos[0], state.videos);
      }
      if (id === "dual_watch") {
        openDualWatchTab();
      }
      if (id === "current_tab") {
        openVideoTab();
      }
    }

    function escapeHtml(value) {
      return String(value).replace(/[&<>"']/g, ch => ({
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        '"': "&quot;",
        "'": "&#039;"
      })[ch]);
    }

    function bookmarkStorageKey(video) {
      return `video-review-center:markers:${video?.id || "unknown"}`;
    }

    function defaultBookmarks() {
      return {
        a: { seconds: null },
        b: { seconds: null }
      };
    }

    function loadBookmarks(video) {
      if (!video?.long_timeline?.length) return defaultBookmarks();
      try {
        const parsed = JSON.parse(localStorage.getItem(bookmarkStorageKey(video)) || "{}");
        const normalized = defaultBookmarks();
        for (const key of ["a", "b"]) {
          const raw = parsed?.[key] || {};
          normalized[key] = {
            seconds: Number.isFinite(raw.seconds) ? raw.seconds : null
          };
        }
        return normalized;
      } catch (err) {
        return defaultBookmarks();
      }
    }

    function saveBookmarks(video, bookmarks) {
      if (!video?.long_timeline?.length) return;
      localStorage.setItem(bookmarkStorageKey(video), JSON.stringify(bookmarks));
    }

    $("playFiltered").addEventListener("click", () => {
      if (state.videos.length) playVideo(state.videos[0], state.videos);
    });
    $("prevClip").addEventListener("click", () => playQueue(state.queueIndex - 1));
    $("nextClip").addEventListener("click", () => playQueue(state.queueIndex + 1));
    $("openCurrentTab").addEventListener("click", () => openVideoTab());
    $("openCurrentDualWatch").addEventListener("click", () => openDualWatchTab());
    $("openMarkerTab").addEventListener("click", () => openMarkerTab());
    $("openDualWatch").addEventListener("click", () => openDualWatchTab());
    $("openEventMap").addEventListener("click", () => {
      window.open("/events", "_blank", "noopener");
    });
    $("refresh").addEventListener("click", async () => {
      await refreshData({ force: true });
    });
    $("openLongBuilder").addEventListener("click", () => {
      window.open("/long-builder", "_blank", "noopener");
    });
    $("openEdgeFrameBuilder").addEventListener("click", () => {
      window.open("/edge-frame-builder", "_blank", "noopener");
    });

    kindEl.addEventListener("change", () => {
      syncScopedFilters({ keepCurrent: false });
      loadVideos();
    });
    groupEl.addEventListener("change", () => {
      syncSubgroupOptions();
      loadVideos();
    });
    subgroupEl.addEventListener("change", loadVideos);
    dateEl.addEventListener("change", loadVideos);
    timeModeEl.addEventListener("change", () => {
      renderClips();
      if (state.current) $("nowPlaying").textContent = `${clipLabel(state.current)} · ${state.current.name}`;
    });
    playbackRateEl.addEventListener("change", () => applyPlaybackRate());
    searchEl.addEventListener("input", () => {
      clearTimeout(searchEl._timer);
      searchEl._timer = setTimeout(loadVideos, 160);
    });
    document.addEventListener("visibilitychange", () => {
      if (document.visibilityState === "visible") {
        refreshData();
      }
    });

    (async function boot() {
      try {
        await loadSummary();
        if ((state.summary.counts.flip || 0) > 0) kindEl.value = "flip";
        await loadVideos();
        state.refreshTimer = setInterval(() => {
          refreshData();
        }, AUTO_REFRESH_MS);
      } catch (err) {
        clipListEl.innerHTML = `<div class="empty">${escapeHtml(err.message)}</div>`;
      }
    })();
  </script>
</body>
</html>"""


EVENT_MAP_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Capture Event Map</title>
  <style>
    :root {
      --bg: #f5f1e8;
      --ink: #171717;
      --muted: #69655b;
      --line: #d5ccbb;
      --panel: #fffdf8;
      --accent: #176b5a;
      --warn: #af4b38;
      --soft: #e9e2d2;
      --ok: #dcefe3;
      --shadow: 0 10px 26px rgba(39, 32, 20, 0.12);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: ui-sans-serif, "Avenir Next", "Segoe UI", sans-serif;
      min-height: 100vh;
      overflow-x: hidden;
      overflow-y: auto;
    }
    header {
      display: flex;
      justify-content: space-between;
      gap: 16px;
      align-items: flex-start;
      padding: 18px 22px;
      border-bottom: 1px solid var(--line);
      background: rgba(255,253,248,0.9);
      position: sticky;
      top: 0;
      z-index: 2;
      backdrop-filter: blur(10px);
    }
    h1 { margin: 0 0 5px; font-size: 24px; }
    button, input, select, textarea { font: inherit; }
    button, select, input, textarea {
      border: 1px solid var(--line);
      border-radius: 6px;
      background: white;
      min-height: 36px;
      padding: 8px 10px;
    }
    textarea { width: 100%; min-height: 80px; resize: vertical; }
    button { cursor: pointer; }
    button.primary { background: var(--accent); border-color: var(--accent); color: white; }
    main { padding: 18px 22px 30px; display: grid; gap: 14px; }
    .muted { color: var(--muted); font-size: 13px; }
    .toolbar, .filters, .stats { display: flex; gap: 8px; flex-wrap: wrap; align-items: center; }
    .filters {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: var(--shadow);
      padding: 12px;
    }
    .filters label {
      display: grid;
      gap: 5px;
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0;
    }
    .filters select { min-width: 150px; }
    .filters input { min-width: 260px; }
    .stat {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
      padding: 10px 12px;
      min-width: 130px;
    }
    .stat strong { display: block; font-size: 22px; }
    .analytics-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(270px, 1fr));
      gap: 10px;
    }
    .metric-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
      gap: 8px;
    }
    .metric {
      border: 1px solid #ece6d8;
      border-radius: 6px;
      padding: 8px;
      background: #fffdf8;
      min-width: 0;
    }
    .metric strong { display: block; font-size: 18px; overflow-wrap: anywhere; }
    .chips { display: flex; flex-wrap: wrap; gap: 5px; margin-top: 8px; }
    .mini-table { margin-top: 8px; max-height: 230px; overflow: auto; }
    .session-list { display: grid; gap: 10px; }
    details.session {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: var(--shadow);
      overflow: hidden;
    }
    summary {
      cursor: pointer;
      display: grid;
      grid-template-columns: minmax(120px, 0.5fr) minmax(260px, 1.2fr) minmax(180px, 0.8fr) minmax(180px, 0.8fr);
      gap: 10px;
      padding: 12px;
      align-items: center;
      list-style: none;
    }
    summary::-webkit-details-marker { display: none; }
    .capture-id { font-size: 18px; font-weight: 800; color: var(--accent); }
    .pill {
      display: inline-flex;
      align-items: center;
      width: max-content;
      min-height: 22px;
      border-radius: 999px;
      background: var(--soft);
      padding: 2px 8px;
      font-size: 12px;
    }
    .pill.complete { background: var(--ok); }
    .pill.open, .pill.interrupted_by_next_start { background: #ffe2d9; color: #7a2516; }
    .event-tags { display: flex; gap: 5px; flex-wrap: wrap; }
    .event-links {
      display: flex;
      gap: 5px;
      flex-wrap: wrap;
      margin-top: 4px;
    }
    .event-link {
      display: inline-flex;
      align-items: center;
      width: max-content;
      min-height: 24px;
      padding: 2px 8px;
      border: 1px solid var(--line);
      border-radius: 999px;
      background: var(--soft);
      color: var(--ink);
      font-size: 12px;
      text-decoration: none;
    }
    .body {
      border-top: 1px solid var(--line);
      padding: 12px;
      display: grid;
      grid-template-columns: minmax(320px, 0.9fr) minmax(360px, 1fr);
      gap: 12px;
    }
    .box {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: white;
      padding: 10px;
      min-width: 0;
    }
    .box-title { font-weight: 800; margin-bottom: 8px; }
    .annotation {
      grid-column: 1 / -1;
      display: grid;
      gap: 8px;
    }
    .annotation-grid {
      display: grid;
      grid-template-columns: minmax(200px, 0.45fr) minmax(280px, 1fr) auto;
      gap: 8px;
      align-items: start;
    }
    .link-button {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      width: max-content;
      min-height: 26px;
      padding: 3px 8px;
      border: 1px solid var(--line);
      border-radius: 999px;
      background: var(--soft);
      color: var(--ink);
      font-size: 12px;
      text-decoration: none;
      white-space: nowrap;
    }
    .availability {
      display: inline-flex;
      align-items: center;
      width: max-content;
      border-radius: 999px;
      padding: 2px 7px;
      font-size: 12px;
      background: #ffe2d9;
      color: #7a2516;
    }
    .availability.yes {
      background: var(--ok);
      color: var(--ink);
    }
    table { width: 100%; border-collapse: collapse; font-size: 13px; }
    th, td { text-align: left; border-bottom: 1px solid #ece6d8; padding: 6px; vertical-align: top; }
    th { color: var(--muted); font-size: 12px; font-weight: 700; }
    td.path { overflow-wrap: anywhere; }
    .empty {
      border: 1px dashed var(--line);
      border-radius: 8px;
      padding: 24px;
      text-align: center;
      color: var(--muted);
      background: rgba(255,255,255,0.5);
    }
    @media (max-width: 920px) {
      header { position: relative; }
      summary { grid-template-columns: 1fr; }
      .body { grid-template-columns: 1fr; }
      .annotation-grid { grid-template-columns: 1fr; }
      .filters label, .filters select, .filters input { width: 100%; min-width: 0; }
    }
  </style>
</head>
<body>
  <header>
    <div>
      <h1>Capture Event Map</h1>
      <div class="muted">Command log joined with remote video CSV by Capture_ID and clip timestamp.</div>
    </div>
    <div class="toolbar">
      <button id="openReview">Review Center</button>
      <button id="exportHtml">Export HTML</button>
      <button id="refresh" class="primary">Refresh</button>
    </div>
  </header>

  <main>
    <section class="stats">
      <div class="stat"><strong id="sessionCount">0</strong><span class="muted">sessions</span></div>
      <div class="stat"><strong id="remoteVideoCount">0</strong><span class="muted">remote videos</span></div>
      <div class="stat"><strong id="linkedVideoCount">0</strong><span class="muted">linked clips</span></div>
      <div class="stat"><strong id="totalCommandCount">0</strong><span class="muted">commands</span></div>
      <div class="stat"><strong id="pumpStartCount">0</strong><span class="muted">pump starts</span></div>
      <div class="stat"><strong id="totalClipDuration">0s</strong><span class="muted">clip time est.</span></div>
      <div class="stat"><strong id="pumpDuration">0s</strong><span class="muted">pump time</span></div>
    </section>

    <section class="filters">
      <label>Capture ID
        <select id="captureId"></select>
      </label>
      <label>Event Type
        <select id="eventType"></select>
      </label>
      <label>Status
        <select id="status"></select>
      </label>
      <label>Time Display
        <select id="timeMode">
          <option value="thai">Thailand GMT+7</option>
          <option value="utc">UTC</option>
        </select>
      </label>
      <label>Search
        <input id="search" placeholder="file, capture id, argument">
      </label>
    </section>

    <div id="sourceInfo" class="muted"></div>
    <section id="analyticsGrid" class="analytics-grid"></section>
    <section id="sessions" class="session-list"></section>
  </main>

  <script>
    const state = {
      summary: null,
      sessions: [],
      exportData: window.__EVENT_EXPORT_DATA || null,
      focusSessionId: new URLSearchParams(window.location.search).get("session_id") || ""
    };
    const $ = (id) => document.getElementById(id);

    async function getJson(url) {
      const res = await fetch(url);
      if (!res.ok) throw new Error(await res.text());
      return res.json();
    }

    async function postJson(url, payload) {
      if (state.exportData) {
        throw new Error("This exported HTML is read-only. Open the live /events page to save notes.");
      }
      const res = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(data.error || "Request failed");
      return data;
    }

    function option(value, label) {
      const opt = document.createElement("option");
      opt.value = value;
      opt.textContent = label;
      return opt;
    }

    function escapeHtml(value) {
      return String(value ?? "").replace(/[&<>"']/g, ch => ({
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        '"': "&quot;",
        "'": "&#039;"
      })[ch]);
    }

    function activeTimeMode() {
      return $("timeMode").value || "thai";
    }

    function formatTime(value) {
      if (!value) return "";
      const date = new Date(`${value}Z`);
      if (Number.isNaN(date.getTime())) return value;
      if (activeTimeMode() === "utc") {
        const parts = new Intl.DateTimeFormat("en-CA", {
          timeZone: "UTC",
          year: "numeric",
          month: "2-digit",
          day: "2-digit",
          hour: "2-digit",
          minute: "2-digit",
          second: "2-digit",
          hour12: false
        }).formatToParts(date);
        const values = Object.fromEntries(parts.map(part => [part.type, part.value]));
        return `${values.year}-${values.month}-${values.day} ${values.hour}:${values.minute}:${values.second} UTC`;
      }
      const parts = new Intl.DateTimeFormat("en-CA", {
        timeZone: "Asia/Bangkok",
        year: "numeric",
        month: "2-digit",
        day: "2-digit",
        hour: "2-digit",
        minute: "2-digit",
        second: "2-digit",
        hour12: false
      }).formatToParts(date);
      const values = Object.fromEntries(parts.map(part => [part.type, part.value]));
      return `${values.year}-${values.month}-${values.day} ${values.hour}:${values.minute}:${values.second} GMT+7`;
    }

    function durationText(seconds) {
      if (seconds === null || seconds === undefined) return "open";
      const value = Number(seconds);
      if (!Number.isFinite(value)) return "";
      const min = Math.floor(value / 60);
      const sec = Math.round(value % 60);
      return min ? `${min}m ${sec}s` : `${sec}s`;
    }

    function longDurationText(seconds) {
      const value = Number(seconds || 0);
      if (!Number.isFinite(value) || value <= 0) return "0s";
      const days = Math.floor(value / 86400);
      const hours = Math.floor((value % 86400) / 3600);
      const minutes = Math.floor((value % 3600) / 60);
      const sec = Math.round(value % 60);
      const parts = [];
      if (days) parts.push(`${days}d`);
      if (hours) parts.push(`${hours}h`);
      if (minutes) parts.push(`${minutes}m`);
      if (!parts.length || sec) parts.push(`${sec}s`);
      return parts.slice(0, 3).join(" ");
    }

    function numberText(value, digits = 0) {
      const number = Number(value || 0);
      if (!Number.isFinite(number)) return "0";
      return number.toLocaleString(undefined, { maximumFractionDigits: digits });
    }

    function sizeText(bytes) {
      const value = Number(bytes || 0);
      if (!Number.isFinite(value) || value <= 0) return "0 B";
      const units = ["B", "KB", "MB", "GB", "TB"];
      let size = value;
      let unit = 0;
      while (size >= 1024 && unit < units.length - 1) {
        size /= 1024;
        unit += 1;
      }
      return `${size.toLocaleString(undefined, { maximumFractionDigits: unit ? 2 : 0 })} ${units[unit]}`;
    }

    function metric(label, value, sub = "") {
      return `<div class="metric"><strong>${escapeHtml(value)}</strong><span class="muted">${escapeHtml(label)}</span>${sub ? `<div class="muted">${escapeHtml(sub)}</div>` : ""}</div>`;
    }

    function fillFilters(summary) {
      const captureCurrent = $("captureId").value || "all";
      const eventCurrent = $("eventType").value || "all";
      const statusCurrent = $("status").value || "all";

      $("captureId").replaceChildren(
        option("all", "All capture IDs"),
        ...(summary.capture_ids || []).map(value => option(value, value))
      );
      $("eventType").replaceChildren(
        option("all", "All events"),
        ...(summary.event_types || []).map(value => option(value, value))
      );
      $("status").replaceChildren(
        option("all", "All statuses"),
        ...(summary.statuses || []).map(value => option(value, value))
      );
      $("captureId").value = (summary.capture_ids || []).includes(captureCurrent) ? captureCurrent : "all";
      $("eventType").value = (summary.event_types || []).includes(eventCurrent) ? eventCurrent : "all";
      $("status").value = (summary.statuses || []).includes(statusCurrent) ? statusCurrent : "all";
    }

    function params() {
      const values = new URLSearchParams({
        capture_id: $("captureId").value || "all",
        event_type: $("eventType").value || "all",
        status: $("status").value || "all",
        q: $("search").value || ""
      });
      if (state.focusSessionId) values.set("session_id", state.focusSessionId);
      return values;
    }

    function filteredExportData() {
      if (!state.exportData) return null;
      const captureId = $("captureId").value || "all";
      const eventType = $("eventType").value || "all";
      const status = $("status").value || "all";
      const query = ($("search").value || "").trim().toLowerCase();
      let sessions = state.exportData.sessions || [];
      if (state.focusSessionId) {
        sessions = sessions.filter(session => session.id === state.focusSessionId);
      }
      if (captureId !== "all") {
        sessions = sessions.filter(session => session.capture_id === captureId);
      }
      if (eventType !== "all") {
        sessions = sessions.filter(session => session.command_counts?.[eventType] || (session.event_types || []).includes(eventType));
      }
      if (status !== "all") {
        sessions = sessions.filter(session => session.status === status);
      }
      if (query) {
        sessions = sessions.filter(session =>
          String(session.capture_id || "").toLowerCase().includes(query)
          || (session.events || []).some(event => String(event.arguments || "").toLowerCase().includes(query))
          || (session.videos || []).some(video => String(video.file_name || "").toLowerCase().includes(query))
        );
      }
      return { ...state.exportData, sessions };
    }

    async function loadData() {
      const data = filteredExportData() || await getJson(`/api/event-sessions?${params()}`);
      state.summary = data.summary;
      state.sessions = data.sessions || [];
      $("sessionCount").textContent = state.summary.session_count || 0;
      $("remoteVideoCount").textContent = state.summary.remote_video_count || 0;
      $("linkedVideoCount").textContent = state.summary.session_video_count || 0;
      $("totalCommandCount").textContent = numberText(state.summary.analytics?.commands?.total || state.summary.session_command_event_count || 0);
      $("pumpStartCount").textContent = numberText(state.summary.analytics?.commands?.pump_start_count || state.summary.session_pump_start_count || 0);
      $("totalClipDuration").textContent = longDurationText(state.summary.session_clip_seconds || 0);
      $("pumpDuration").textContent = longDurationText(state.summary.analytics?.commands?.pump_requested_seconds || state.summary.session_pump_requested_seconds || 0);
      $("sourceInfo").textContent = `command: ${state.summary.command_csv || "-"} · remote videos: ${state.summary.remote_video_csv || "-"} · CSV scanned: ${state.summary.analytics?.csv_file_count || 0}`;
      if (state.exportData?.exported_at) {
        $("sourceInfo").textContent += ` · exported: ${state.exportData.exported_at}`;
      }
      fillFilters(state.summary);
      renderGlobalAnalytics();
      renderSessions();
    }

    function renderGlobalAnalytics() {
      const analytics = state.summary?.analytics || {};
      const commands = analytics.commands || {};
      const remote = analytics.remote_videos || {};
      const flip = analytics.flip_report || {};
      const flipTotal = flip.total || {};
      const telemetry = analytics.telemetry || [];
      const longRanges = analytics.long_ranges || {};
      const csvFiles = analytics.csv_files || [];
      const commandChips = (commands.top_commands || []).map(item => `<span class="pill">${escapeHtml(item.name)} ${numberText(item.count)}</span>`).join("");
      const partChips = (remote.top_parts || []).slice(0, 8).map(item => `<span class="pill">${escapeHtml(item.part)} ${numberText(item.count)}</span>`).join("");
      const sourceRows = (flip.top_sources || []).map(source => `
        <tr>
          <td class="path">${escapeHtml(source.source_root)}</td>
          <td>${numberText(source.unique_files)}</td>
          <td>${escapeHtml(source.remote_size || sizeText(source.remote_size_bytes))}</td>
        </tr>
      `).join("");
      const telemetryRows = telemetry.map(item => `
        <tr>
          <td>${escapeHtml(item.name)}</td>
          <td>${numberText(item.row_count)}</td>
          <td>${escapeHtml(numberText(item.min, 3))}</td>
          <td>${escapeHtml(numberText(item.max, 3))}</td>
          <td>${escapeHtml(numberText(item.avg, 3))}</td>
          <td>${numberText(item.nonzero_count)}</td>
        </tr>
      `).join("");
      const fileChips = csvFiles.slice(0, 18).map(file => `<span class="pill">${escapeHtml(file.kind)} · ${escapeHtml(file.rel_path)}</span>`).join("");
      const moreFiles = csvFiles.length > 18 ? `<span class="muted">+${csvFiles.length - 18} more CSVs</span>` : "";

      $("analyticsGrid").innerHTML = `
        <div class="box">
          <div class="box-title">Commands + Pump</div>
          <div class="metric-grid">
            ${metric("deduped commands", numberText(commands.total))}
            ${metric("command CSVs", numberText(commands.csv_count))}
            ${metric("PumpStart", numberText(commands.pump_start_count))}
            ${metric("PumpStop", numberText(commands.pump_stop_count))}
            ${metric("pump requested", longDurationText(commands.pump_requested_seconds))}
            ${metric("pump observed", longDurationText(commands.pump_observed_seconds))}
            ${metric("TorchControl", numberText(commands.torch_command_count))}
          </div>
          <div class="chips">${commandChips}</div>
        </div>
        <div class="box">
          <div class="box-title">Clip Catalog</div>
          <div class="metric-grid">
            ${metric("linked clips", numberText(state.summary?.session_video_count))}
            ${metric("linked clip time est.", longDurationText(state.summary?.session_clip_seconds))}
            ${metric("linked remote size", `${numberText(state.summary?.session_remote_size_mb, 2)} MB`)}
            ${metric("all remote CSV clips", numberText(remote.total))}
            ${metric("all remote CSV size", sizeText(remote.total_size_bytes))}
            ${metric("capture parts", numberText(remote.part_count))}
            ${metric("to_long rows", numberText(longRanges.row_count), `${numberText(longRanges.estimated_source_clips)} source clips`)}
          </div>
          <div class="chips">${partChips}</div>
        </div>
        <div class="box">
          <div class="box-title">Flip Missing Source</div>
          <div class="metric-grid">
            ${metric("expected remote", numberText(flipTotal.expected_remote_files))}
            ${metric("present in flip", numberText(flipTotal.present_in_flip))}
            ${metric("missing from flip", numberText(flipTotal.missing_from_flip))}
            ${metric("found in sources", numberText(flipTotal.missing_found_in_sources))}
            ${metric("still missing", numberText(flipTotal.still_missing))}
            ${metric("still missing size", flipTotal.still_missing_size || sizeText(flipTotal.still_missing_size_bytes))}
          </div>
          <div class="mini-table">
            <table><thead><tr><th>Source</th><th>Files</th><th>Size</th></tr></thead><tbody>${sourceRows}</tbody></table>
          </div>
        </div>
        <div class="box">
          <div class="box-title">Telemetry CSV</div>
          <div class="mini-table">
            <table><thead><tr><th>Metric</th><th>Rows</th><th>Min</th><th>Max</th><th>Avg</th><th>Nonzero</th></tr></thead><tbody>${telemetryRows}</tbody></table>
          </div>
        </div>
        <div class="box">
          <div class="box-title">CSV Sources</div>
          <div class="metric-grid">
            ${metric("CSV files scanned", numberText(analytics.csv_file_count))}
            ${metric("flip report CSVs", numberText(flip.csv_count))}
            ${metric("telemetry exports", numberText(telemetry.length))}
          </div>
          <div class="chips">${fileChips}${moreFiles}</div>
        </div>
      `;
    }

    function eventTags(session) {
      const values = session.event_types || [];
      const tags = values.map(value => `<span class="pill">${escapeHtml(value)}</span>`);
      if ((session.merged_session_count || 1) > 1) {
        tags.unshift(`<span class="pill complete">merged ${escapeHtml(session.merged_session_count)} sessions</span>`);
      }
      if (!tags.length) return `<span class="muted">capture only</span>`;
      return tags.join("");
    }

    function renderCommandTable(events) {
      return `
        <table>
          <thead><tr><th>Time</th><th>Command</th><th>Arguments</th></tr></thead>
          <tbody>
            ${events.map(event => `
              <tr>
                <td>${escapeHtml(formatTime(event.time))}</td>
                <td>${escapeHtml(event.command_type)}</td>
                <td class="path">${escapeHtml(event.arguments)}</td>
              </tr>
            `).join("")}
          </tbody>
        </table>
      `;
    }

    function renderVideoTable(videos) {
      if (!videos.length) return `<div class="empty">No remote videos matched this session window.</div>`;
      return `
        <table>
          <thead><tr><th>Time</th><th>Clip</th><th>Local</th><th>Remote path</th><th>MB</th></tr></thead>
          <tbody>
            ${videos.map(video => `
              <tr>
                <td>${escapeHtml(formatTime(video.time))}</td>
                <td>${escapeHtml(video.file_name)}</td>
                <td>${renderLocalAvailability(video)}</td>
                <td class="path">${escapeHtml(video.remote_path)}</td>
                <td>${escapeHtml(video.size_mb)}</td>
              </tr>
            `).join("")}
          </tbody>
        </table>
      `;
    }

    function renderLocalAvailability(video) {
      const matches = video.local_matches || [];
      if (!video.local_available || !matches.length) {
        return `<span class="availability">Missing</span>`;
      }
      const first = matches[0];
      const more = video.local_match_count > 1 ? ` +${video.local_match_count - 1}` : "";
      return `
        <span class="availability yes">Available</span>
        <a class="link-button" href="${escapeHtml(first.watch_url)}" target="_blank" rel="noopener">Watch${escapeHtml(more)}</a>
      `;
    }

    function renderSessionMetrics(session) {
      const flip = session.flip_report || {};
      const torchRange = session.torch_command_count
        ? `${numberText(session.torch_min, 2)}-${numberText(session.torch_max, 2)} avg ${numberText(session.torch_avg, 2)}`
        : "-";
      return `
        <div class="box session-metrics">
          <div class="box-title">Session Metrics</div>
          <div class="metric-grid">
            ${metric("commands", numberText(session.event_count), session.command_density_per_minute ? `${numberText(session.command_density_per_minute, 2)}/min` : "")}
            ${metric("pump starts", numberText(session.pump_start_count), `${numberText(session.pump_command_count)} pump commands`)}
            ${metric("pump requested", longDurationText(session.pump_requested_seconds), `observed ${longDurationText(session.pump_observed_seconds)}`)}
            ${metric("pump PWM avg", `A ${numberText(session.pump_avg_pwm_a, 1)} / B ${numberText(session.pump_avg_pwm_b, 1)}`)}
            ${metric("clip time est.", longDurationText(session.estimated_clip_seconds), `${numberText(session.clip_seconds_per_video)}s per clip`)}
            ${metric("remote data", `${numberText(session.total_remote_size_mb, 2)} MB`, `${numberText(session.local_missing_count)} missing local`)}
            ${metric("torch commands", numberText(session.torch_command_count), torchRange)}
            ${metric("flip missing part", numberText(flip.still_missing), flip.still_missing_size || "")}
          </div>
        </div>
      `;
    }

    function renderAnnotation(session) {
      return `
        <div class="box annotation" data-session-id="${escapeHtml(session.id)}">
          <div class="box-title">Session Name / Note</div>
          <div class="annotation-grid">
            <input data-field="group-name" value="${escapeHtml(session.group_name || "")}" placeholder="Name for this CaptureStart to CaptureStop session">
            <textarea data-field="group-note" placeholder="Note for this session">${escapeHtml(session.group_note || "")}</textarea>
            <button data-action="save-note" class="primary">Save</button>
          </div>
          <div class="muted">Saved to this session only: ${escapeHtml(session.capture_id)} · ${escapeHtml(formatTime(session.start_time))}.</div>
        </div>
      `;
    }

    function renderSessions() {
      const el = $("sessions");
      if (!state.sessions.length) {
        el.innerHTML = `<div class="empty">No sessions match this filter.</div>`;
        return;
      }

      el.replaceChildren(...state.sessions.map(session => {
        const node = document.createElement("details");
        node.className = "session";
        if (session.id === state.focusSessionId) node.open = true;
        const displayTitle = session.group_name
          ? `${session.capture_id} · ${session.group_name}`
          : session.capture_id;
        node.innerHTML = `
          <summary>
            <div>
              <div class="capture-id">${escapeHtml(displayTitle)}</div>
              <span class="pill ${escapeHtml(session.status)}">${escapeHtml(session.status)}</span>
            </div>
            <div>
              <div><strong>${escapeHtml(formatTime(session.start_time))}</strong></div>
              <div class="muted">duration ${escapeHtml(durationText(session.duration_seconds))} · cameras ${escapeHtml(session.camera_a || "?")}/${escapeHtml(session.camera_b || "?")}</div>
            </div>
            <div>
              <div><strong>${session.video_count}</strong> clips · <strong>${numberText(session.pump_start_count)}</strong> pump</div>
              <div class="muted">${session.local_video_count || 0} local · ${session.event_count} commands · ${longDurationText(session.estimated_clip_seconds)} est.</div>
            </div>
            <div class="event-tags">${eventTags(session)}</div>
          </summary>
          <div class="body">
            ${renderAnnotation(session)}
            ${renderSessionMetrics(session)}
            <div class="box">
              <div class="box-title">Events</div>
              ${renderCommandTable(session.events || [])}
            </div>
            <div class="box">
              <div class="box-title">Remote Clips</div>
              ${renderVideoTable(session.videos || [])}
            </div>
          </div>
        `;
        return node;
      }));
      bindAnnotationButtons();
      if (state.focusSessionId) {
        const focused = Array.from(el.querySelectorAll("details.session")).find((node, index) => {
          return state.sessions[index]?.id === state.focusSessionId;
        });
        if (focused) focused.scrollIntoView({ block: "center" });
      }
    }

    function bindAnnotationButtons() {
      document.querySelectorAll('[data-action="save-note"]').forEach(button => {
        button.addEventListener("click", async () => {
          const box = button.closest("[data-session-id]");
          if (!box) return;
          button.disabled = true;
          const sessionId = box.dataset.sessionId;
          const name = box.querySelector('[data-field="group-name"]').value;
          const note = box.querySelector('[data-field="group-note"]').value;
          try {
            await postJson("/api/event-group-notes", { session_id: sessionId, name, note });
            await loadData();
          } catch (err) {
            alert(err.message);
          } finally {
            button.disabled = false;
          }
        });
      });
    }

    $("refresh").addEventListener("click", loadData);
    $("openReview").addEventListener("click", () => window.open("/", "_blank", "noopener"));
    $("exportHtml").addEventListener("click", () => {
      const query = params().toString();
      window.open(`/events-export.html${query ? `?${query}` : ""}`, "_blank", "noopener");
    });
    ["captureId", "eventType", "status"].forEach(id => $(id).addEventListener("change", () => {
      state.focusSessionId = "";
      loadData();
    }));
    $("timeMode").addEventListener("change", renderSessions);
    $("search").addEventListener("input", () => {
      state.focusSessionId = "";
      clearTimeout($("search")._timer);
      $("search")._timer = setTimeout(loadData, 160);
    });

    loadData().catch(err => {
      $("sessions").innerHTML = `<div class="empty">${escapeHtml(err.message)}</div>`;
    });
  </script>
</body>
</html>"""


LONG_BUILDER_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Long Builder</title>
  <style>
    :root {
      --bg: #f4f2ec;
      --ink: #151515;
      --muted: #6d6a62;
      --line: #d7d0c2;
      --panel: #fffdf7;
      --accent: #1f6f5b;
      --soft: #e8e2d3;
      --shadow: 0 12px 34px rgba(41, 34, 20, 0.12);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background:
        linear-gradient(135deg, rgba(31, 111, 91, 0.10), transparent 35%),
        linear-gradient(315deg, rgba(184, 75, 53, 0.08), transparent 38%),
        var(--bg);
      color: var(--ink);
      font-family: ui-sans-serif, "Avenir Next", "Segoe UI", sans-serif;
    }
    main {
      width: min(980px, calc(100vw - 28px));
      margin: 18px auto 30px;
      display: grid;
      gap: 14px;
    }
    .panel {
      background: rgba(255, 253, 247, 0.9);
      border: 1px solid var(--line);
      border-radius: 16px;
      box-shadow: var(--shadow);
      padding: 18px;
    }
    h1 {
      margin: 0 0 6px;
      font-size: 32px;
      line-height: 1.05;
    }
    .muted {
      color: var(--muted);
      font-size: 14px;
    }
    .grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px 12px;
      margin-top: 18px;
    }
    .field {
      display: grid;
      gap: 6px;
    }
    label {
      font-size: 12px;
      color: var(--muted);
      text-transform: uppercase;
    }
    input, select, textarea, button {
      font: inherit;
    }
    input, select, textarea {
      width: 100%;
      border: 1px solid var(--line);
      background: white;
      border-radius: 12px;
      padding: 12px 14px;
      min-height: 48px;
    }
    textarea {
      min-height: 120px;
      resize: vertical;
    }
    .hidden {
      display: none !important;
    }
    .mode-panel.hidden {
      display: none;
    }
    .callout {
      margin-top: 14px;
      padding: 12px 14px;
      border-radius: 12px;
      border: 1px solid var(--line);
      background: rgba(232, 226, 211, 0.55);
      display: grid;
      gap: 6px;
      font-size: 13px;
      color: var(--muted);
    }
    .span-2 {
      grid-column: 1 / -1;
    }
    .actions {
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      align-items: center;
      margin-top: 16px;
    }
    button {
      border: 1px solid var(--line);
      background: var(--panel);
      color: var(--ink);
      border-radius: 12px;
      padding: 12px 16px;
      cursor: pointer;
    }
    button.primary {
      background: var(--accent);
      border-color: var(--accent);
      color: white;
    }
    details {
      margin-top: 12px;
      border-top: 1px dashed var(--line);
      padding-top: 12px;
    }
    summary {
      cursor: pointer;
      font-weight: 700;
    }
    .tips {
      margin-top: 14px;
      display: grid;
      gap: 8px;
      font-size: 13px;
      color: var(--muted);
    }
    .jobs {
      display: grid;
      gap: 10px;
    }
    .job {
      border: 1px solid var(--line);
      border-radius: 12px;
      background: white;
      padding: 12px;
      display: grid;
      gap: 7px;
    }
    .job-top {
      display: flex;
      justify-content: space-between;
      gap: 10px;
      align-items: start;
    }
    .status {
      display: inline-flex;
      padding: 4px 9px;
      border-radius: 999px;
      background: var(--soft);
      font-size: 12px;
    }
    .status.running { background: #fff1c7; }
    .status.completed { background: #d9f2df; }
    .status.failed { background: #ffd8d2; }
    .logbox {
      border-top: 1px dashed var(--line);
      padding-top: 8px;
      max-height: 180px;
      overflow: auto;
      font-size: 12px;
      display: grid;
      gap: 4px;
    }
    .empty {
      padding: 20px;
      border: 1px dashed var(--line);
      border-radius: 12px;
      text-align: center;
      color: var(--muted);
      background: rgba(255,255,255,0.45);
    }
    @media (max-width: 760px) {
      .grid { grid-template-columns: 1fr; }
      .span-2 { grid-column: auto; }
    }
  </style>
</head>
<body>
  <main>
    <section class="panel">
      <h1>Long Builder</h1>
      <div class="muted">Standalone builder for long videos from `--flip` only. Outputs are written under `event/row_xxx...` next to the folder where clips are found.</div>

      <div class="grid">
        <div class="field">
          <label for="rowNumber">Row Number</label>
          <input id="rowNumber" placeholder="18">
        </div>
        <div class="field">
          <label for="buildMode">Build Mode</label>
          <select id="buildMode">
            <option value="single">Single Range</option>
            <option value="sequence">Sequence Mix</option>
          </select>
        </div>
        <div class="field" id="singleModePrefixField">
          <label for="clipPrefix">Clip Set</label>
          <input id="clipPrefix" placeholder="20260527_153639">
        </div>
        <div class="field" id="singleModeRangeField">
          <label for="fileRange">Range</label>
          <input id="fileRange" placeholder="0-30">
        </div>
        <div class="field span-2 mode-panel" id="singleModePanel">
          <div class="muted">Example output name: `row_018_20260529_053928_027_034`</div>
        </div>
        <div class="field span-2 mode-panel hidden" id="sequenceModePanel">
          <label for="outputName">Output Name (optional)</label>
          <input id="outputName" placeholder="night_mix_cam_a">
          <div class="muted">If set, this becomes the folder/file name, for example `row_018_night_mix_cam_a`.</div>
        </div>
        <div class="field span-2 mode-panel hidden" id="sequenceItemsField">
          <label for="sequenceText">Sequence Items</label>
          <textarea id="sequenceText" placeholder="20260527_153639 0-5&#10;20260529_053928 27-34&#10;double_20260530_010000_012.mp4&#10;cam_a/night_pick_01"></textarea>
        </div>
        <div class="field">
          <label for="requireComplete">Missing Files</label>
          <select id="requireComplete">
            <option value="false">Skip Missing</option>
            <option value="true">Require Complete</option>
          </select>
        </div>
        <div class="field">
          <label for="crf">CRF</label>
          <input id="crf" value="0" placeholder="0">
        </div>
        <div class="field">
          <label for="preset">Preset</label>
          <select id="preset">
            <option value="veryslow">veryslow</option>
            <option value="slower">slower</option>
            <option value="slow">slow</option>
            <option value="medium">medium</option>
            <option value="fast">fast</option>
            <option value="veryfast">veryfast</option>
          </select>
        </div>
      </div>

      <div class="callout">
        <div>`Single Range` keeps the original behavior: one range per job.</div>
        <div>`Sequence Mix` accepts multiple lines. Each line can be a range like `YYYYMMDD_HHMMSS 0-5`, a clip name like `double_...mp4`, or a folder path inside flip to append the whole folder.</div>
      </div>

      <details>
        <summary>Advanced: Paste multiple rows</summary>
        <div class="grid">
          <div class="field">
            <label for="delimiter">Rows Delimiter</label>
            <select id="delimiter">
              <option value="auto">Auto</option>
              <option value="tab">Tab</option>
              <option value="comma">Comma</option>
              <option value="pipe">Pipe</option>
            </select>
          </div>
          <div class="field">
            <label for="rowNumberColumn">Row Number Column</label>
            <input id="rowNumberColumn" placeholder="1-based">
          </div>
          <div class="field">
            <label for="prefixColumn">Prefix Column</label>
            <input id="prefixColumn" placeholder="1-based">
          </div>
          <div class="field">
            <label for="rangeColumn">Range Column</label>
            <input id="rangeColumn" placeholder="1-based">
          </div>
          <div class="field span-2">
            <label for="rowsText">Rows</label>
            <textarea id="rowsText" placeholder="Paste multiple rows here to create several jobs at once"></textarea>
          </div>
        </div>
      </details>

      <div class="actions">
        <button id="createLong" class="primary">Create Long</button>
        <button id="refreshJobs">Refresh Status</button>
        <button id="openReview">Back To Review</button>
      </div>
      <div id="buildStatus" class="muted">Ready to build long video</div>

      <div class="tips">
        <div>`Sequence Mix` is useful for clips that are not contiguous, cross days/times, or need to be combined into one long video.</div>
        <div>Enter one item per line in `Sequence Items`. The system concatenates them from top to bottom.</div>
        <div>For folders, prefer a path relative to `flip` to avoid ambiguous duplicate folder names.</div>
        <div>If `Output Name` is empty, the system generates a name from the first and last clip.</div>
        <div>`Advanced` is only for pasting multiple rows where row number, prefix, and range columns must be mapped.</div>
      </div>
    </section>

    <section class="panel">
      <div style="font-weight:700; margin-bottom:10px;">Build Status</div>
      <div id="jobList" class="jobs"></div>
    </section>
  </main>

  <script>
    const $ = (id) => document.getElementById(id);
    const rowNumberEl = $("rowNumber");
    const buildModeEl = $("buildMode");
    const clipPrefixEl = $("clipPrefix");
    const fileRangeEl = $("fileRange");
    const outputNameEl = $("outputName");
    const sequenceTextEl = $("sequenceText");
    const requireCompleteEl = $("requireComplete");
    const crfEl = $("crf");
    const presetEl = $("preset");
    const delimiterEl = $("delimiter");
    const rowNumberColumnEl = $("rowNumberColumn");
    const prefixColumnEl = $("prefixColumn");
    const rangeColumnEl = $("rangeColumn");
    const rowsTextEl = $("rowsText");
    const buildStatusEl = $("buildStatus");
    const jobListEl = $("jobList");
    const singleModePrefixFieldEl = $("singleModePrefixField");
    const singleModeRangeFieldEl = $("singleModeRangeField");
    const singleModePanelEl = $("singleModePanel");
    const sequenceModePanelEl = $("sequenceModePanel");
    const sequenceItemsFieldEl = $("sequenceItemsField");
    let timer = null;

    async function getJson(url) {
      const res = await fetch(url);
      if (!res.ok) throw new Error(await res.text());
      return res.json();
    }

    async function postJson(url, payload) {
      const res = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(data.error || "Request failed");
      return data;
    }

    function escapeHtml(value) {
      return String(value).replace(/[&<>"']/g, ch => ({
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        '"': "&quot;",
        "'": "&#039;"
      })[ch]);
    }

    function fileName(value) {
      const text = String(value || "");
      const parts = text.split("/");
      return parts[parts.length - 1] || text;
    }

    function syncModeUi() {
      const isSequence = buildModeEl.value === "sequence";
      singleModePrefixFieldEl.classList.toggle("hidden", isSequence);
      singleModeRangeFieldEl.classList.toggle("hidden", isSequence);
      singleModePanelEl.classList.toggle("hidden", isSequence);
      sequenceModePanelEl.classList.toggle("hidden", !isSequence);
      sequenceItemsFieldEl.classList.toggle("hidden", !isSequence);
    }

    function renderJobs(jobs) {
      if (!jobs.length) {
        jobListEl.innerHTML = '<div class="empty">No long-video jobs yet</div>';
        return;
      }
      jobListEl.replaceChildren(...jobs.map(job => {
        const el = document.createElement("article");
        el.className = "job";
        const outputs = (job.outputs || []).map(item => escapeHtml(fileName(item.output_video))).join("<br>");
        const missing = (job.missing || []).map(line => escapeHtml(line)).join("<br>");
        const logs = (job.logs || []).slice().reverse().map(line => `<div>${escapeHtml(line)}</div>`).join("");
        el.innerHTML = `
          <div class="job-top">
            <div>
              <div style="font-weight:700;">${escapeHtml(job.progress?.message || job.id)}</div>
              <div class="muted">${escapeHtml(job.id)} · rows ${job.rows_total || 0}</div>
            </div>
            <span class="status ${escapeHtml(job.status)}">${escapeHtml(job.status)}</span>
          </div>
          ${outputs ? `<div><strong>Outputs</strong><div class="muted">${outputs}</div></div>` : ""}
          ${missing ? `<div><strong>Missing</strong><div class="muted">${missing}</div></div>` : ""}
          ${job.error ? `<div><strong>Error</strong><div class="muted">${escapeHtml(job.error)}</div></div>` : ""}
          <div class="logbox">${logs}</div>
        `;
        return el;
      }));
    }

    async function loadJobs() {
      const data = await getJson("/api/long-jobs");
      const jobs = data.jobs || [];
      const active = jobs.find(job => job.status === "queued" || job.status === "running");
      if (active) {
        buildStatusEl.textContent = active.progress?.message || "Building long video...";
      } else if (jobs[0]) {
        buildStatusEl.textContent = jobs[0].progress?.message || `Latest: ${jobs[0].status}`;
      } else {
        buildStatusEl.textContent = "Ready to build long video";
      }
      renderJobs(jobs);
    }

    async function createLong() {
      buildStatusEl.textContent = "Submitting build request...";
      try {
        const job = await postJson("/api/long-jobs", {
          rowNumber: rowNumberEl.value,
          buildMode: buildModeEl.value,
          clipPrefix: clipPrefixEl.value,
          fileRange: fileRangeEl.value,
          outputName: outputNameEl.value,
          sequenceText: sequenceTextEl.value,
          requireComplete: requireCompleteEl.value,
          crf: crfEl.value,
          preset: presetEl.value,
          delimiter: delimiterEl.value,
          rowNumberColumn: rowNumberColumnEl.value,
          prefixColumn: prefixColumnEl.value,
          rangeColumn: rangeColumnEl.value,
          rowsText: rowsTextEl.value
        });
        buildStatusEl.textContent = job.progress?.message || "Queued.";
        await loadJobs();
      } catch (err) {
        buildStatusEl.textContent = err.message;
      }
    }

    $("createLong").addEventListener("click", createLong);
    $("refreshJobs").addEventListener("click", loadJobs);
    buildModeEl.addEventListener("change", syncModeUi);
    $("openReview").addEventListener("click", () => {
      window.location.href = "/";
    });

    (async function boot() {
      syncModeUi();
      await loadJobs().catch(() => {});
      timer = setInterval(() => {
        loadJobs().catch(() => {});
      }, 2000);
    })();
  </script>
</body>
</html>"""


EDGE_FRAME_BUILDER_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Edge Frame Builder</title>
  <style>
    :root {
      --bg: #f4f2ec;
      --ink: #151515;
      --muted: #6d6a62;
      --line: #d7d0c2;
      --panel: #fffdf7;
      --accent: #1f6f5b;
      --shadow: 0 12px 34px rgba(41, 34, 20, 0.12);
      font-family: "Segoe UI", system-ui, sans-serif;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: linear-gradient(180deg, #f8f4eb 0%, #efe6d6 100%);
      color: var(--ink);
      min-height: 100vh;
    }
    main {
      max-width: 1180px;
      margin: 0 auto;
      padding: 28px 18px 40px;
      display: grid;
      gap: 18px;
    }
    .panel {
      background: rgba(255,255,255,0.82);
      backdrop-filter: blur(8px);
      border: 1px solid rgba(255,255,255,0.7);
      border-radius: 20px;
      box-shadow: var(--shadow);
      padding: 18px;
    }
    h1 {
      margin: 0 0 10px;
      font-size: 30px;
    }
    .muted {
      color: var(--muted);
      font-size: 14px;
    }
    .grid {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 12px;
      margin-top: 18px;
    }
    .field {
      display: grid;
      gap: 6px;
    }
    .span-2 {
      grid-column: span 2;
    }
    .span-3 {
      grid-column: span 3;
    }
    label {
      font-size: 13px;
      color: var(--muted);
      font-weight: 600;
    }
    input, select, textarea, button {
      width: 100%;
      font: inherit;
      border-radius: 12px;
      border: 1px solid var(--line);
      padding: 11px 12px;
      background: white;
      color: var(--ink);
    }
    textarea {
      min-height: 180px;
      resize: vertical;
    }
    .actions {
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      margin-top: 16px;
    }
    button {
      cursor: pointer;
      width: auto;
      min-width: 150px;
    }
    button.primary {
      background: var(--accent);
      border-color: var(--accent);
      color: white;
    }
    details {
      margin-top: 12px;
      border-top: 1px dashed var(--line);
      padding-top: 12px;
    }
    summary {
      cursor: pointer;
      font-weight: 700;
    }
    .tips {
      margin-top: 14px;
      display: grid;
      gap: 8px;
      font-size: 13px;
      color: var(--muted);
    }
    .jobs {
      display: grid;
      gap: 10px;
    }
    .job {
      border: 1px solid var(--line);
      border-radius: 12px;
      background: white;
      padding: 12px;
      display: grid;
      gap: 7px;
    }
    .job-top {
      display: flex;
      justify-content: space-between;
      gap: 10px;
      align-items: start;
    }
    .status {
      display: inline-flex;
      padding: 4px 9px;
      border-radius: 999px;
      background: var(--soft);
      font-size: 12px;
    }
    .status.running { background: #fff1c7; }
    .status.completed { background: #d9f2df; }
    .status.failed { background: #ffd8d2; }
    .logbox {
      border-top: 1px dashed var(--line);
      padding-top: 8px;
      max-height: 180px;
      overflow: auto;
      font-size: 12px;
      display: grid;
      gap: 4px;
    }
    .empty {
      padding: 20px;
      border: 1px dashed var(--line);
      border-radius: 12px;
      text-align: center;
      color: var(--muted);
      background: rgba(255,255,255,0.45);
    }
    @media (max-width: 760px) {
      .grid { grid-template-columns: 1fr; }
      .span-2, .span-3 { grid-column: auto; }
    }
  </style>
</head>
<body>
  <main>
    <section class="panel">
      <h1>Edge Frame Builder</h1>
      <div class="muted">Build a slideshow from the first/last frame of each clip using the same row + clip set + range structure as long videos. Outputs are written under `event/row_xxx_..._edge_frames`.</div>

      <div class="grid">
        <div class="field">
          <label for="rowNumber">Row Number</label>
          <input id="rowNumber" placeholder="18">
        </div>
        <div class="field">
          <label for="clipPrefix">Clip Set</label>
          <input id="clipPrefix" placeholder="20260527_153639">
        </div>
        <div class="field">
          <label for="fileRange">Range</label>
          <input id="fileRange" placeholder="27-34">
        </div>
        <div class="field">
          <label for="fps">FPS</label>
          <input id="fps" value="30" placeholder="30">
        </div>
        <div class="field">
          <label for="frameHoldSeconds">Hold Seconds</label>
          <input id="frameHoldSeconds" value="1" placeholder="1">
        </div>
        <div class="field">
          <label for="defaultClipSeconds">Default Clip Seconds</label>
          <input id="defaultClipSeconds" value="60" placeholder="60">
        </div>
        <div class="field">
          <label for="requireComplete">Missing Files</label>
          <select id="requireComplete">
            <option value="false">Skip Missing</option>
            <option value="true">Require Complete</option>
          </select>
        </div>
        <div class="field span-3">
          <div class="muted">Example folder name: `row_018_20260527_153639_027_034_edge_frames`</div>
        </div>
      </div>

      <details>
        <summary>Advanced: Paste multiple rows</summary>
        <div class="grid">
          <div class="field">
            <label for="crf">CRF</label>
            <input id="crf" value="0" placeholder="0">
            <div class="muted">Lower values are sharper but create larger files.</div>
          </div>
          <div class="field">
            <label for="preset">Preset</label>
            <select id="preset">
              <option value="veryslow">veryslow</option>
              <option value="slower">slower</option>
              <option value="slow">slow</option>
              <option value="medium">medium</option>
              <option value="fast">fast</option>
              <option value="veryfast">veryfast</option>
            </select>
            <div class="muted">veryslow takes the longest but compresses most carefully.</div>
          </div>
          <div class="field">
            <label for="jpgQuality">JPG Quality</label>
            <input id="jpgQuality" value="2" placeholder="2">
            <div class="muted">Only affects the legacy image-extraction mode.</div>
          </div>
          <div class="field">
            <label for="delimiter">Rows Delimiter</label>
            <select id="delimiter">
              <option value="auto">Auto</option>
              <option value="tab">Tab</option>
              <option value="comma">Comma</option>
              <option value="pipe">Pipe</option>
            </select>
          </div>
          <div class="field">
            <label for="rowNumberColumn">Row Number Column</label>
            <input id="rowNumberColumn" placeholder="1-based">
          </div>
          <div class="field">
            <label for="prefixColumn">Prefix Column</label>
            <input id="prefixColumn" placeholder="1-based">
          </div>
          <div class="field">
            <label for="rangeColumn">Range Column</label>
            <input id="rangeColumn" placeholder="1-based">
          </div>
          <div class="field span-3">
            <label for="rowsText">Rows</label>
            <textarea id="rowsText" placeholder="Paste multiple rows here to create several jobs at once"></textarea>
          </div>
        </div>
      </details>

      <div class="actions">
        <button id="createEdgeFrames" class="primary">Create Edge Frames</button>
        <button id="refreshJobs">Refresh Status</button>
        <button id="openReview">Back To Review</button>
      </div>
      <div id="buildStatus" class="muted">Ready to build edge frames</div>

      <div class="tips">
        <div>The three core fields are still `Row Number`, `Clip Set`, and `Range`.</div>
        <div>`Hold Seconds` controls how long the first and last frame stay on screen per side. Example: 1 = first 1 second + last 1 second.</div>
        <div>Outputs include a video, `frames/`, `note.txt`, and `manifest.json` so they can be reviewed in Video Review Center.</div>
      </div>
    </section>

    <section class="panel">
      <div style="font-weight:700; margin-bottom:10px;">Build Status</div>
      <div id="jobList" class="jobs"></div>
    </section>
  </main>

  <script>
    const $ = (id) => document.getElementById(id);
    const rowNumberEl = $("rowNumber");
    const clipPrefixEl = $("clipPrefix");
    const fileRangeEl = $("fileRange");
    const fpsEl = $("fps");
    const frameHoldSecondsEl = $("frameHoldSeconds");
    const defaultClipSecondsEl = $("defaultClipSeconds");
    const requireCompleteEl = $("requireComplete");
    const crfEl = $("crf");
    const presetEl = $("preset");
    const jpgQualityEl = $("jpgQuality");
    const delimiterEl = $("delimiter");
    const rowNumberColumnEl = $("rowNumberColumn");
    const prefixColumnEl = $("prefixColumn");
    const rangeColumnEl = $("rangeColumn");
    const rowsTextEl = $("rowsText");
    const buildStatusEl = $("buildStatus");
    const jobListEl = $("jobList");
    let timer = null;

    async function getJson(url) {
      const res = await fetch(url);
      if (!res.ok) throw new Error(await res.text());
      return res.json();
    }

    async function postJson(url, payload) {
      const res = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(data.error || "Request failed");
      return data;
    }

    function escapeHtml(value) {
      return String(value).replace(/[&<>"']/g, ch => ({
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        '"': "&quot;",
        "'": "&#039;"
      })[ch]);
    }

    function fileName(value) {
      const text = String(value || "");
      const parts = text.split("/");
      return parts[parts.length - 1] || text;
    }

    function renderJobs(jobs) {
      if (!jobs.length) {
        jobListEl.innerHTML = '<div class="empty">No edge-frame jobs yet</div>';
        return;
      }
      jobListEl.replaceChildren(...jobs.map(job => {
        const el = document.createElement("article");
        el.className = "job";
        const outputs = (job.outputs || []).map(item => escapeHtml(fileName(item.output_video))).join("<br>");
        const missing = (job.missing || []).map(line => escapeHtml(line)).join("<br>");
        const logs = (job.logs || []).slice().reverse().map(line => `<div>${escapeHtml(line)}</div>`).join("");
        el.innerHTML = `
          <div class="job-top">
            <div>
              <div style="font-weight:700;">${escapeHtml(job.progress?.message || job.id)}</div>
              <div class="muted">${escapeHtml(job.id)} · rows ${job.rows_total || 0}</div>
            </div>
            <span class="status ${escapeHtml(job.status)}">${escapeHtml(job.status)}</span>
          </div>
          ${outputs ? `<div><strong>Outputs</strong><div class="muted">${outputs}</div></div>` : ""}
          ${missing ? `<div><strong>Missing</strong><div class="muted">${missing}</div></div>` : ""}
          ${job.error ? `<div><strong>Error</strong><div class="muted">${escapeHtml(job.error)}</div></div>` : ""}
          <div class="logbox">${logs}</div>
        `;
        return el;
      }));
    }

    async function loadJobs() {
      const data = await getJson("/api/edge-frame-jobs");
      const jobs = data.jobs || [];
      const active = jobs.find(job => job.status === "queued" || job.status === "running");
      if (active) {
        buildStatusEl.textContent = active.progress?.message || "Building edge frames...";
      } else if (jobs[0]) {
        buildStatusEl.textContent = jobs[0].progress?.message || `Latest: ${jobs[0].status}`;
      } else {
        buildStatusEl.textContent = "Ready to build edge frames";
      }
      renderJobs(jobs);
    }

    async function createEdgeFrames() {
      buildStatusEl.textContent = "Submitting build request...";
      try {
        const job = await postJson("/api/edge-frame-jobs", {
          rowNumber: rowNumberEl.value,
          clipPrefix: clipPrefixEl.value,
          fileRange: fileRangeEl.value,
          fps: fpsEl.value,
          framesPerImage: "",
          frameHoldSeconds: frameHoldSecondsEl.value,
          defaultClipSeconds: defaultClipSecondsEl.value,
          requireComplete: requireCompleteEl.value,
          crf: crfEl.value,
          preset: presetEl.value,
          jpgQuality: jpgQualityEl.value,
          delimiter: delimiterEl.value,
          rowNumberColumn: rowNumberColumnEl.value,
          prefixColumn: prefixColumnEl.value,
          rangeColumn: rangeColumnEl.value,
          rowsText: rowsTextEl.value
        });
        buildStatusEl.textContent = job.progress?.message || "Queued.";
        await loadJobs();
      } catch (err) {
        buildStatusEl.textContent = err.message;
      }
    }

    $("createEdgeFrames").addEventListener("click", createEdgeFrames);
    $("refreshJobs").addEventListener("click", loadJobs);
    $("openReview").addEventListener("click", () => {
      window.location.href = "/";
    });

    (async function boot() {
      await loadJobs().catch(() => {});
      timer = setInterval(() => {
        loadJobs().catch(() => {});
      }, 2000);
    })();
  </script>
</body>
</html>"""


DUAL_WATCH_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Two-Up Watch</title>
  <style>
    :root {
      --bg: #f4f2ec;
      --ink: #151515;
      --muted: #6d6a62;
      --line: #d7d0c2;
      --panel: #fffdf7;
      --accent: #1f6f5b;
      --accent-2: #b84b35;
      --shadow: 0 12px 34px rgba(41, 34, 20, 0.12);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: ui-sans-serif, "Avenir Next", "Segoe UI", sans-serif;
    }
    header {
      position: sticky;
      top: 0;
      z-index: 5;
      display: grid;
      gap: 10px;
      padding: 12px 14px;
      background: rgba(255, 253, 247, 0.94);
      border-bottom: 1px solid var(--line);
      backdrop-filter: blur(12px);
    }
    .topbar {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: center;
    }
    h1 {
      margin: 0;
      font-size: 20px;
      line-height: 1.1;
    }
    button, input, select {
      font: inherit;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: var(--panel);
      color: var(--ink);
      min-height: 36px;
      padding: 8px 10px;
    }
    input, select { width: 100%; }
    button { cursor: pointer; width: max-content; }
    button.primary {
      background: var(--accent);
      border-color: var(--accent);
      color: white;
    }
    .actions {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      align-items: center;
    }
    .actions button,
    .actions select {
      width: auto;
    }
    .muted {
      color: var(--muted);
      font-size: 13px;
    }
    .selectors {
      display: grid;
      grid-template-columns: minmax(260px, 1fr) minmax(260px, 1fr);
      gap: 10px;
    }
    .field {
      display: grid;
      gap: 6px;
      min-width: 0;
    }
    label {
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0;
    }
    main {
      width: 100vw;
      display: grid;
      grid-template-rows: repeat(2, minmax(0, 1fr));
      gap: 0;
      height: var(--viewer-height, calc(100vh - 156px));
      overflow: hidden;
    }
    .slot {
      display: grid;
      grid-template-rows: minmax(0, 1fr) auto;
      background: #000;
      min-height: 0;
      overflow: hidden;
    }
    .slot + .slot {
      border-top: 1px solid #111;
    }
    video {
      display: block;
      width: 100vw;
      height: 100%;
      min-height: 0;
      background: #000;
      object-fit: contain;
    }
    .custom-controls {
      display: grid;
      grid-template-columns: auto 1fr auto auto auto auto auto;
      gap: 8px;
      align-items: center;
      padding: 9px 14px;
      background: #050505;
      color: white;
      border-top: 1px solid #222;
    }
    .custom-controls button {
      background: #161616;
      border-color: #343434;
      color: white;
    }
    .custom-controls input[type="range"] {
      width: 100%;
      accent-color: var(--accent);
    }
    .custom-controls .step-control {
      display: inline-flex;
      gap: 6px;
      align-items: center;
      color: #ddd;
      font-size: 13px;
      white-space: nowrap;
    }
    .custom-controls .step-control select {
      min-height: 36px;
      padding: 7px 8px;
      border: 1px solid #343434;
      border-radius: 6px;
      background: #111;
      color: white;
      font: inherit;
    }
    .custom-controls .time-readout {
      min-width: 132px;
      color: #ddd;
      font-size: 13px;
      text-align: right;
      font-variant-numeric: tabular-nums;
    }
    .empty {
      padding: 34px;
      color: var(--muted);
      text-align: center;
      background: rgba(255,255,255,0.45);
    }
    @media (max-width: 880px) {
      .topbar { align-items: flex-start; flex-direction: column; }
      .selectors { grid-template-columns: 1fr; }
      .custom-controls {
        grid-template-columns: auto 1fr auto;
      }
      .custom-controls .step-control,
      .custom-controls .time-readout {
        grid-column: 1 / -1;
      }
    }
  </style>
</head>
<body>
  <header>
    <div class="topbar">
      <div>
        <h1>Two-Up Watch</h1>
        <div id="info" class="muted">Loading clips...</div>
      </div>
      <div class="actions">
        <select id="timeMode">
          <option value="gmt">GMT from file</option>
          <option value="thai">GMT+7 Thailand</option>
        </select>
        <select id="playbackRate">
          <option value="0.5">0.5x</option>
          <option value="1">1x</option>
          <option value="1.5">1.5x</option>
          <option value="2">2x</option>
          <option value="3">3x</option>
          <option value="4">4x</option>
          <option value="8">8x</option>
          <option value="16">16x</option>
        </select>
        <button id="syncPlay" class="primary">Sync Play</button>
        <button id="syncPause">Sync Pause</button>
        <button id="syncToTop">Sync To Top</button>
        <button id="swapVideos">Swap</button>
      </div>
    </div>
    <div class="selectors">
      <div class="field">
        <label for="topSearch">Top Clip</label>
        <input id="topSearch" placeholder="Search top clip">
        <select id="topSelect"></select>
      </div>
      <div class="field">
        <label for="bottomSearch">Bottom Clip</label>
        <input id="bottomSearch" placeholder="Search bottom clip">
        <select id="bottomSelect"></select>
      </div>
    </div>
  </header>

  <main>
    <section id="topSlot" class="slot"></section>
    <section id="bottomSlot" class="slot"></section>
  </main>

  <script>
    const params = new URLSearchParams(location.search);
    const seedIds = (params.get("ids") || params.get("id") || "").split(",").filter(Boolean);
    const timeModeEl = document.getElementById("timeMode");
    const playbackRateEl = document.getElementById("playbackRate");
    const topSelectEl = document.getElementById("topSelect");
    const bottomSelectEl = document.getElementById("bottomSelect");
    const topSearchEl = document.getElementById("topSearch");
    const bottomSearchEl = document.getElementById("bottomSearch");
    const slots = {
      top: {
        select: topSelectEl,
        search: topSearchEl,
        root: document.getElementById("topSlot"),
        video: null,
        data: null
      },
      bottom: {
        select: bottomSelectEl,
        search: bottomSearchEl,
        root: document.getElementById("bottomSlot"),
        video: null,
        data: null
      }
    };
    let videos = [];
    timeModeEl.value = params.get("time") || "gmt";
    playbackRateEl.value = params.get("speed") || "1";

    function updateViewerHeight() {
      const header = document.querySelector("header");
      const headerHeight = header ? header.offsetHeight : 0;
      const available = Math.max(0, window.innerHeight - headerHeight);
      document.documentElement.style.setProperty("--viewer-height", `${available}px`);
    }

    async function getJson(url) {
      const res = await fetch(url);
      if (!res.ok) throw new Error(await res.text());
      return res.json();
    }

    function escapeHtml(value) {
      return String(value).replace(/[&<>"']/g, ch => ({
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        '"': "&quot;",
        "'": "&#039;"
      })[ch]);
    }

    function activePlaybackRate() {
      const rate = Number(playbackRateEl.value);
      return Number.isFinite(rate) && rate > 0 ? rate : 1;
    }

    function applyPlaybackRate() {
      for (const slot of Object.values(slots)) {
        if (slot.video) slot.video.playbackRate = activePlaybackRate();
      }
    }

    function groupLabel(video) {
      if (!video) return "";
      return video.subgroup && video.subgroup !== "root"
        ? `${video.group}/${video.subgroup}`
        : video.group;
    }

    function formatTime(video) {
      if (!video?.timestamp) return "unknown time";
      if (timeModeEl.value === "thai") {
        const date = new Date(`${video.timestamp}Z`);
        if (!Number.isNaN(date.getTime())) {
          const parts = new Intl.DateTimeFormat("en-CA", {
            timeZone: "Asia/Bangkok",
            year: "numeric",
            month: "2-digit",
            day: "2-digit",
            hour: "2-digit",
            minute: "2-digit",
            second: "2-digit",
            hour12: false
          }).formatToParts(date);
          const values = Object.fromEntries(parts.map(part => [part.type, part.value]));
          return `${values.year}-${values.month}-${values.day} ${values.hour}:${values.minute}:${values.second} GMT+7`;
        }
      }
      return `${video.date} ${video.time} GMT`;
    }

    function optionLabel(video) {
      return `${formatTime(video)} | ${video.name} | ${groupLabel(video)} | ${video.kind}`;
    }

    function videoMatches(video, query) {
      if (!query) return true;
      const haystack = [
        video.name,
        video.rel_path,
        video.group,
        video.subgroup,
        video.date,
        video.time,
        video.kind
      ].join(" ").toLowerCase();
      return haystack.includes(query);
    }

    function renderSelect(slotName, preferredId = "") {
      const slot = slots[slotName];
      const current = preferredId || slot.select.value;
      const query = slot.search.value.trim().toLowerCase();
      const selectedVideo = videos.find(video => video.id === current);
      const matches = videos.filter(video => videoMatches(video, query)).slice(0, 300);
      const list = selectedVideo && !matches.some(video => video.id === selectedVideo.id)
        ? [selectedVideo, ...matches]
        : matches;
      const opts = list.map(video => {
        const opt = document.createElement("option");
        opt.value = video.id;
        opt.textContent = optionLabel(video);
        return opt;
      });
      slot.select.replaceChildren(...opts);
      if (list.some(video => video.id === current)) {
        slot.select.value = current;
      } else if (list.length) {
        slot.select.value = list[0].id;
      }
      document.getElementById("info").textContent =
        `${videos.length} clips indexed. Selectors show up to 300 matches while searching.`;
    }

    function formatDuration(seconds) {
      if (!Number.isFinite(seconds) || seconds < 0) return "0:00";
      const whole = Math.floor(seconds);
      const minutes = Math.floor(whole / 60);
      const secs = String(whole % 60).padStart(2, "0");
      return `${minutes}:${secs}`;
    }

    function renderSlot(slotName) {
      const slot = slots[slotName];
      const videoData = videos.find(video => video.id === slot.select.value);
      slot.data = videoData || null;
      if (!videoData) {
        slot.root.innerHTML = `<div class="empty">No ${escapeHtml(slotName)} clip selected.</div>`;
        slot.video = null;
        return;
      }
      slot.root.innerHTML = `
        <video playsinline loop src="${videoData.url}"></video>
        <div class="custom-controls">
          <button type="button" data-control="play">Play</button>
          <input type="range" data-control="seek" min="0" max="1000" value="0" step="1">
          <span class="time-readout" data-control="time">0:00 / 0:00</span>
          <label class="step-control">Step
            <select data-control="step">
              <option value="1">1s</option>
              <option value="2">2s</option>
              <option value="5">5s</option>
              <option value="10">10s</option>
              <option value="20">20s</option>
              <option value="30">30s</option>
              <option value="60">60s</option>
            </select>
          </label>
          <button type="button" data-control="back">Back</button>
          <button type="button" data-control="forward">Forward</button>
          <button type="button" data-control="full">Full</button>
        </div>
      `;
      setupSlotControls(slotName);
    }

    function clampSeconds(value, duration) {
      if (!Number.isFinite(value)) return 0;
      if (!Number.isFinite(duration) || duration <= 0) return Math.max(0, value);
      return Math.max(0, Math.min(duration, value));
    }

    function setupSlotControls(slotName) {
      const slot = slots[slotName];
      const item = slot.root;
      const video = item.querySelector("video");
      const playBtn = item.querySelector('[data-control="play"]');
      const seek = item.querySelector('[data-control="seek"]');
      const time = item.querySelector('[data-control="time"]');
      const stepInput = item.querySelector('[data-control="step"]');
      const fullBtn = item.querySelector('[data-control="full"]');
      const backBtn = item.querySelector('[data-control="back"]');
      const forwardBtn = item.querySelector('[data-control="forward"]');
      slot.video = video;
      video.playbackRate = activePlaybackRate();

      const activeStep = () => {
        const step = Math.trunc(Number(stepInput.value));
        const bounded = Number.isFinite(step) && step > 0 ? Math.min(step, 60) : 1;
        stepInput.value = String(bounded);
        return bounded;
      };

      const update = () => {
        const duration = video.duration || 0;
        seek.value = duration ? String(Math.round((video.currentTime / duration) * 1000)) : "0";
        time.textContent = `${formatDuration(video.currentTime)} / ${formatDuration(duration)}`;
        playBtn.textContent = video.paused ? "Play" : "Pause";
      };

      const nudge = (direction) => {
        const duration = video.duration || 0;
        video.currentTime = clampSeconds(video.currentTime + (direction * activeStep()), duration);
        update();
      };

      playBtn.addEventListener("click", () => {
        if (video.paused) video.play().catch(() => {});
        else video.pause();
      });
      seek.addEventListener("input", () => {
        if (video.duration) video.currentTime = (Number(seek.value) / 1000) * video.duration;
      });
      backBtn.addEventListener("click", () => nudge(-1));
      forwardBtn.addEventListener("click", () => nudge(1));
      fullBtn.addEventListener("click", () => {
        if (document.fullscreenElement) document.exitFullscreen();
        else item.requestFullscreen?.();
      });
      video.addEventListener("click", () => playBtn.click());
      video.addEventListener("loadedmetadata", update);
      video.addEventListener("timeupdate", update);
      video.addEventListener("play", update);
      video.addEventListener("pause", update);
      update();
    }

    function updateUrl() {
      const ids = [topSelectEl.value, bottomSelectEl.value].filter(Boolean);
      const next = new URLSearchParams({
        ids: ids.join(","),
        time: timeModeEl.value,
        speed: playbackRateEl.value
      });
      history.replaceState(null, "", `/dual-watch?${next.toString()}`);
    }

    function renderSelectedSlots() {
      renderSlot("top");
      renderSlot("bottom");
      updateUrl();
      document.title = `Two-Up Watch (${topSelectEl.selectedOptions[0]?.textContent || "Top"} vs ${bottomSelectEl.selectedOptions[0]?.textContent || "Bottom"})`;
    }

    function sync(method) {
      for (const slot of Object.values(slots)) {
        if (!slot.video) continue;
        slot.video.playbackRate = activePlaybackRate();
        if (method === "play") slot.video.play().catch(() => {});
        if (method === "pause") slot.video.pause();
      }
    }

    function toggleSyncPlayback() {
      const activeVideos = Object.values(slots).map(slot => slot.video).filter(Boolean);
      if (!activeVideos.length) return;
      const shouldPlay = activeVideos.every(video => video.paused);
      sync(shouldPlay ? "play" : "pause");
    }

    function syncToTop() {
      const top = slots.top.video;
      const bottom = slots.bottom.video;
      if (!top || !bottom) return;
      bottom.currentTime = clampSeconds(top.currentTime, bottom.duration || 0);
      bottom.playbackRate = top.playbackRate;
    }

    function swapVideos() {
      const topId = topSelectEl.value;
      topSelectEl.value = bottomSelectEl.value;
      bottomSelectEl.value = topId;
      renderSelectedSlots();
    }

    function initSelection() {
      const seededTop = videos.some(video => video.id === seedIds[0]) ? seedIds[0] : "";
      const topId = seededTop || videos[0]?.id || "";
      const seededBottom = videos.some(video => video.id === seedIds[1]) ? seedIds[1] : "";
      const bottomId = seededBottom || videos.find(video => video.id !== topId)?.id || "";
      renderSelect("top", topId);
      renderSelect("bottom", bottomId);
      topSelectEl.value = topId;
      bottomSelectEl.value = bottomId;
      renderSelectedSlots();
    }

    topSearchEl.addEventListener("input", () => renderSelect("top"));
    bottomSearchEl.addEventListener("input", () => renderSelect("bottom"));
    topSelectEl.addEventListener("change", renderSelectedSlots);
    bottomSelectEl.addEventListener("change", renderSelectedSlots);
    timeModeEl.addEventListener("change", renderSelectedSlots);
    playbackRateEl.addEventListener("change", () => {
      applyPlaybackRate();
      updateUrl();
    });
    document.getElementById("syncPlay").addEventListener("click", () => sync("play"));
    document.getElementById("syncPause").addEventListener("click", () => sync("pause"));
    document.getElementById("syncToTop").addEventListener("click", syncToTop);
    document.getElementById("swapVideos").addEventListener("click", swapVideos);
    document.addEventListener("keydown", event => {
      const tag = document.activeElement?.tagName;
      if (tag === "INPUT" || tag === "SELECT" || tag === "TEXTAREA") return;
      if (event.code === "Space" || event.key === " ") {
        event.preventDefault();
        toggleSyncPlayback();
      }
    });
    window.addEventListener("resize", updateViewerHeight);

    (async function boot() {
      try {
        updateViewerHeight();
        const data = await getJson("/api/videos?kind=all&group=all&subgroup=all&date=all&q=");
        videos = data.videos || [];
        initSelection();
        requestAnimationFrame(updateViewerHeight);
      } catch (err) {
        document.getElementById("info").textContent = err.message;
        document.getElementById("topSlot").innerHTML = `<div class="empty">${escapeHtml(err.message)}</div>`;
      }
    })();
  </script>
</body>
</html>"""


TAB_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Video Compare Tab</title>
  <style>
    :root {
      --bg: #f4f2ec;
      --ink: #151515;
      --muted: #6d6a62;
      --line: #d7d0c2;
      --panel: #fffdf7;
      --accent: #1f6f5b;
      --accent-2: #b84b35;
      --shadow: 0 12px 34px rgba(41, 34, 20, 0.12);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background:
        linear-gradient(135deg, rgba(31, 111, 91, 0.10), transparent 35%),
        linear-gradient(315deg, rgba(184, 75, 53, 0.10), transparent 38%),
        var(--bg);
      color: var(--ink);
      font-family: ui-sans-serif, "Avenir Next", "Segoe UI", sans-serif;
    }
    header {
      position: sticky;
      top: 0;
      z-index: 5;
      display: flex;
      justify-content: space-between;
      gap: 16px;
      align-items: center;
      padding: 14px 18px;
      background: rgba(255, 253, 247, 0.90);
      border-bottom: 1px solid var(--line);
      backdrop-filter: blur(12px);
    }
    h1 {
      margin: 0;
      font-size: 20px;
      line-height: 1.1;
    }
    button, select {
      font: inherit;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: var(--panel);
      color: var(--ink);
      min-height: 36px;
      padding: 8px 10px;
    }
    button.primary {
      background: var(--accent);
      border-color: var(--accent);
      color: white;
    }
    .actions {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      align-items: center;
    }
    .muted {
      color: var(--muted);
      font-size: 13px;
    }
    main {
      width: min(1180px, calc(100vw - 28px));
      margin: 18px auto 28px;
      display: grid;
      gap: 12px;
    }
    .item {
      display: grid;
      grid-template-columns: minmax(340px, 0.98fr) minmax(280px, 1fr);
      gap: 12px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: rgba(255, 253, 247, 0.90);
      box-shadow: var(--shadow);
      padding: 12px;
      align-items: start;
    }
    video {
      display: block;
      width: 100%;
      background: #111;
      border-radius: 6px;
      aspect-ratio: 16 / 9;
    }
    .meta {
      display: grid;
      gap: 8px;
      min-width: 0;
    }
    .time {
      color: var(--accent);
      font-size: 16px;
      font-weight: 800;
      overflow-wrap: anywhere;
    }
    .name {
      font-weight: 700;
      overflow-wrap: anywhere;
    }
    .tag {
      display: inline-flex;
      width: max-content;
      padding: 3px 8px;
      border-radius: 999px;
      background: #e8e2d3;
      font-size: 12px;
    }
    .empty {
      padding: 28px;
      border: 1px dashed var(--line);
      border-radius: 8px;
      background: rgba(255,255,255,0.48);
      color: var(--muted);
      text-align: center;
    }
    main.single-video {
      width: 100vw;
      margin: 0 0 28px;
      gap: 0;
    }
    main.single-video .item {
      display: block;
      border: 0;
      border-radius: 0;
      box-shadow: none;
      padding: 0;
      background: transparent;
    }
    main.single-video video {
      display: block;
      width: 100vw;
      max-height: calc(100vh - 142px);
      border-radius: 0;
      aspect-ratio: auto;
      object-fit: contain;
      background: #000;
    }
    .custom-controls {
      display: none;
    }
    main.single-video .custom-controls {
      display: grid;
      grid-template-columns: auto 1fr auto auto auto;
      gap: 10px;
      align-items: center;
      padding: 10px 14px;
      background: #050505;
      color: white;
      border-bottom: 1px solid #222;
    }
    .custom-controls button {
      background: #161616;
      border-color: #343434;
      color: white;
    }
    .custom-controls input[type="range"] {
      width: 100%;
      accent-color: var(--accent);
    }
    .custom-controls .step-control {
      display: inline-flex;
      gap: 6px;
      align-items: center;
      color: #ddd;
      font-size: 13px;
      white-space: nowrap;
    }
    .custom-controls .step-control select {
      min-height: 36px;
      padding: 7px 8px;
      border: 1px solid #343434;
      border-radius: 6px;
      background: #111;
      color: white;
      font: inherit;
    }
    .custom-controls .time-readout {
      min-width: 132px;
      color: #ddd;
      font-size: 13px;
      text-align: right;
      font-variant-numeric: tabular-nums;
    }
    .custom-controls .real-time {
      grid-column: 1 / -1;
      color: #f5f5f5;
      font-size: 15px;
      font-weight: 700;
      overflow-wrap: anywhere;
    }
    .bookmark-panel {
      display: none;
    }
    main.single-video.marker-mode .bookmark-panel {
      display: grid;
      gap: 8px;
      padding: 10px 14px 12px;
      background: #0a0a0a;
      color: white;
      border-bottom: 1px solid #222;
    }
    .bookmark-head {
      display: flex;
      justify-content: flex-start;
      gap: 10px;
      align-items: center;
      flex-wrap: wrap;
    }
    .bookmark-head .muted {
      color: #bbb;
      font-size: 12px;
    }
    .bookmark-actions {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
    }
    .bookmark-row {
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto auto;
      gap: 8px;
      align-items: center;
    }
    .bookmark-track {
      position: relative;
      height: 22px;
      border-radius: 999px;
      background: linear-gradient(90deg, rgba(31, 111, 91, 0.28), rgba(184, 75, 53, 0.32));
      border: 1px solid #2d2d2d;
      overflow: hidden;
    }
    .bookmark-track::after {
      content: "";
      position: absolute;
      inset: 0;
      background:
        linear-gradient(90deg, rgba(255,255,255,0.06), rgba(255,255,255,0.01)),
        repeating-linear-gradient(
          90deg,
          transparent 0,
          transparent calc(10% - 1px),
          rgba(255,255,255,0.08) calc(10% - 1px),
          rgba(255,255,255,0.08) 10%
        );
      pointer-events: none;
    }
    .bookmark-marker {
      position: absolute;
      top: 50%;
      width: 12px;
      height: 12px;
      margin-left: -6px;
      border: 0;
      border-radius: 999px;
      transform: translateY(-50%);
      cursor: pointer;
      box-shadow: 0 0 0 2px rgba(10, 10, 10, 0.9);
    }
    .bookmark-marker.marker-a {
      background: #e1f36b;
    }
    .bookmark-marker.marker-b {
      background: #ff8e67;
    }
    .bookmark-pill {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      width: 22px;
      height: 22px;
      border-radius: 999px;
      font-size: 12px;
      font-weight: 800;
      color: #111;
    }
    .bookmark-pill.marker-a {
      background: #e1f36b;
    }
    .bookmark-pill.marker-b {
      background: #ff8e67;
    }
    .bookmark-meta {
      color: #d8d8d8;
      font-size: 11px;
      line-height: 1.25;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
      min-width: 0;
    }
    main.single-video .meta {
      padding: 12px 18px 18px;
      background: rgba(255, 253, 247, 0.90);
      border-bottom: 1px solid var(--line);
    }
    @media (max-width: 820px) {
      header { align-items: flex-start; flex-direction: column; }
      .item { grid-template-columns: 1fr; }
      .bookmark-row { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <header>
    <div>
      <h1 id="pageTitle">Compare Tab</h1>
      <div id="info" class="muted">Loading...</div>
    </div>
    <div class="actions">
      <select id="timeMode">
        <option value="gmt">GMT from file</option>
        <option value="thai">GMT+7 Thailand</option>
      </select>
      <select id="playbackRate">
        <option value="0.5">0.5x</option>
        <option value="1">1x</option>
        <option value="1.5">1.5x</option>
        <option value="2">2x</option>
        <option value="3">3x</option>
        <option value="4">4x</option>
        <option value="8">8x</option>
        <option value="16">16x</option>
      </select>
      <button id="syncPlay" class="primary">Sync Play</button>
      <button id="syncPause">Sync Pause</button>
    </div>
  </header>
  <main id="list"></main>

  <script>
    const params = new URLSearchParams(location.search);
    const ids = (params.get("ids") || params.get("id") || "").split(",").filter(Boolean);
    const mode = location.pathname === "/markers" ? "markers" : "default";
    const listEl = document.getElementById("list");
    const timeModeEl = document.getElementById("timeMode");
    const playbackRateEl = document.getElementById("playbackRate");
    const pageTitleEl = document.getElementById("pageTitle");
    timeModeEl.value = params.get("time") || "gmt";
    playbackRateEl.value = params.get("speed") || "1";
    let videos = [];

    async function getJson(url) {
      const res = await fetch(url);
      if (!res.ok) throw new Error(await res.text());
      return res.json();
    }

    function escapeHtml(value) {
      return String(value).replace(/[&<>"']/g, ch => ({
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        '"': "&quot;",
        "'": "&#039;"
      })[ch]);
    }

    function bookmarkStorageKey(video) {
      return `video-review-center:markers:${video?.id || "unknown"}`;
    }

    function defaultBookmarks() {
      return {
        a: { seconds: null },
        b: { seconds: null }
      };
    }

    function loadBookmarks(video) {
      if (!video?.long_timeline?.length) return defaultBookmarks();
      try {
        const parsed = JSON.parse(localStorage.getItem(bookmarkStorageKey(video)) || "{}");
        const normalized = defaultBookmarks();
        for (const key of ["a", "b"]) {
          const raw = parsed?.[key] || {};
          normalized[key] = {
            seconds: Number.isFinite(raw.seconds) ? raw.seconds : null
          };
        }
        return normalized;
      } catch (err) {
        return defaultBookmarks();
      }
    }

    function saveBookmarks(video, bookmarks) {
      if (!video?.long_timeline?.length) return;
      localStorage.setItem(bookmarkStorageKey(video), JSON.stringify(bookmarks));
    }

    function activePlaybackRate() {
      const rate = Number(playbackRateEl.value);
      return Number.isFinite(rate) && rate > 0 ? rate : 1;
    }

    function applyPlaybackRate() {
      const rate = activePlaybackRate();
      document.querySelectorAll("video").forEach(video => {
        video.playbackRate = rate;
      });
    }

    function formatTime(video) {
      if (!video.timestamp) return "unknown time";
      if (timeModeEl.value === "thai") {
        const date = new Date(`${video.timestamp}Z`);
        if (!Number.isNaN(date.getTime())) {
          const parts = new Intl.DateTimeFormat("en-CA", {
            timeZone: "Asia/Bangkok",
            year: "numeric",
            month: "2-digit",
            day: "2-digit",
            hour: "2-digit",
            minute: "2-digit",
            second: "2-digit",
            hour12: false
          }).formatToParts(date);
          const values = Object.fromEntries(parts.map(part => [part.type, part.value]));
          return `${values.year}-${values.month}-${values.day} ${values.hour}:${values.minute}:${values.second} GMT+7`;
        }
      }
      return `${video.date} ${video.time} GMT`;
    }

    function render() {
      pageTitleEl.textContent = mode === "markers" ? "Marker Tab" : "Compare Tab";
      document.title = mode === "markers" ? "Video Marker Tab" : "Video Compare Tab";
      document.getElementById("info").textContent = `${videos.length} clips`;
      listEl.classList.toggle("single-video", videos.length === 1);
      listEl.classList.toggle("marker-mode", mode === "markers" && videos.length === 1);
      if (!videos.length) {
        listEl.innerHTML = `<div class="empty">No videos were selected for this tab.</div>`;
        return;
      }
      const single = videos.length === 1;
      listEl.replaceChildren(...videos.map(video => {
        const item = document.createElement("section");
        item.className = "item";
        item.innerHTML = `
          <video ${single ? "" : "controls"} playsinline loop src="${video.url}"></video>
          ${single ? `
            <div class="custom-controls">
              <button type="button" data-control="play">Play</button>
              <input type="range" data-control="seek" min="0" max="1000" value="0" step="1">
              <span class="time-readout" data-control="time">0:00 / 0:00</span>
              <label class="step-control">Step
                <select data-control="step">
                  <option value="1">1s</option>
                  <option value="2">2s</option>
                  <option value="5">5s</option>
                  <option value="10">10s</option>
                  <option value="20">20s</option>
                  <option value="30">30s</option>
                  <option value="60">60s</option>
                </select>
              </label>
              <button type="button" data-control="full">Full</button>
              <div class="real-time" data-control="real">Real: unavailable</div>
            </div>
            <div class="bookmark-panel" data-control="bookmark-panel">
              <div class="bookmark-head">
                <strong>Markers</strong>
                <div class="muted" data-control="bookmark-note">Set two quick jump points for this long video.</div>
                <div class="bookmark-actions">
                  <button type="button" data-control="set-a">Set A</button>
                  <button type="button" data-control="set-b">Set B</button>
                  <button type="button" data-control="reset-bookmarks">Reset</button>
                </div>
              </div>
              <div class="bookmark-row">
                <div class="bookmark-track" data-control="bookmark-track">
                  <button type="button" class="bookmark-marker marker-a" data-marker="a" hidden title="Jump to A"></button>
                  <button type="button" class="bookmark-marker marker-b" data-marker="b" hidden title="Jump to B"></button>
                </div>
                <button type="button" data-control="jump-a"><span class="bookmark-pill marker-a">A</span></button>
                <div class="bookmark-meta" data-control="meta-a">A: not set</div>
              </div>
              <div class="bookmark-row">
                <div class="bookmark-track" data-control="bookmark-track-2">
                  <button type="button" class="bookmark-marker marker-a" data-marker="a-ghost" hidden disabled aria-hidden="true"></button>
                  <button type="button" class="bookmark-marker marker-b" data-marker="b-ghost" hidden disabled aria-hidden="true"></button>
                </div>
                <button type="button" data-control="jump-b"><span class="bookmark-pill marker-b">B</span></button>
                <div class="bookmark-meta" data-control="meta-b">B: not set</div>
              </div>
            </div>
          ` : ""}
          <div class="meta">
            <div class="time">${escapeHtml(formatTime(video))}</div>
            <div class="name">${escapeHtml(video.name)}</div>
            <div class="muted">${escapeHtml(video.group)} · part ${video.part}</div>
            <div class="muted">${escapeHtml(video.rel_path)}</div>
            <span class="tag">${escapeHtml(video.kind)}</span>
          </div>
        `;
        return item;
      }));
      applyPlaybackRate();
      if (single) setupSingleVideoControls();
    }

    function formatDuration(seconds) {
      if (!Number.isFinite(seconds) || seconds < 0) return "0:00";
      const whole = Math.floor(seconds);
      const minutes = Math.floor(whole / 60);
      const secs = String(whole % 60).padStart(2, "0");
      return `${minutes}:${secs}`;
    }

    function parseManifestDate(value) {
      if (!value) return null;
      const [datePart, timePartRaw = "00:00:00"] = String(value).split(" ");
      const [timePart, fraction = ""] = timePartRaw.split(".");
      const millis = fraction ? `.${fraction.slice(0, 3).padEnd(3, "0")}` : "";
      const date = new Date(`${datePart}T${timePart}${millis}Z`);
      return Number.isNaN(date.getTime()) ? null : date;
    }

    function formatRealDate(date) {
      if (!(date instanceof Date) || Number.isNaN(date.getTime())) return "unknown time";
      if (timeModeEl.value === "thai") {
        const parts = new Intl.DateTimeFormat("en-CA", {
          timeZone: "Asia/Bangkok",
          year: "numeric",
          month: "2-digit",
          day: "2-digit",
          hour: "2-digit",
          minute: "2-digit",
          second: "2-digit",
          hour12: false
        }).formatToParts(date);
        const values = Object.fromEntries(parts.map(part => [part.type, part.value]));
        return `${values.year}-${values.month}-${values.day} ${values.hour}:${values.minute}:${values.second} GMT+7`;
      }
      return `${date.toISOString().slice(0, 19).replace("T", " ")} GMT`;
    }

    function clampSeconds(value, duration) {
      if (!Number.isFinite(value)) return null;
      if (!Number.isFinite(duration) || duration <= 0) return Math.max(0, value);
      return Math.max(0, Math.min(duration, value));
    }

    function realTimelinePoint(videoData, outputSeconds) {
      const timeline = videoData?.long_timeline || [];
      if (!timeline.length) return null;
      const segment = timeline.find(item =>
        outputSeconds >= item.output_start_seconds && outputSeconds < item.output_end_seconds
      ) || timeline[timeline.length - 1];
      const sourceStart = parseManifestDate(segment.source_start);
      if (!sourceStart) return null;
      const offsetSeconds = Math.max(0, outputSeconds - segment.output_start_seconds);
      return {
        segment,
        date: new Date(sourceStart.getTime() + (offsetSeconds * 1000))
      };
    }

    function setupSingleVideoControls() {
      const item = listEl.querySelector(".item");
      const video = item?.querySelector("video");
      const playBtn = item?.querySelector('[data-control="play"]');
      const seek = item?.querySelector('[data-control="seek"]');
      const time = item?.querySelector('[data-control="time"]');
      const real = item?.querySelector('[data-control="real"]');
      const stepInput = item?.querySelector('[data-control="step"]');
      const fullBtn = item?.querySelector('[data-control="full"]');
      const bookmarkPanel = item?.querySelector('[data-control="bookmark-panel"]');
      const bookmarkNote = item?.querySelector('[data-control="bookmark-note"]');
      const videoData = videos[0];
      if (!item || !video || !playBtn || !seek || !time || !real || !stepInput || !fullBtn) return;
      const markerMode = mode === "markers";
      const hasTimeline = Boolean(videoData?.long_timeline?.length);
      const bookmarks = loadBookmarks(videoData);

      const bookmarkEls = {
        a: {
          markers: [
            item.querySelector('[data-marker="a"]'),
            item.querySelector('[data-marker="a-ghost"]')
          ].filter(Boolean),
          jump: item.querySelector('[data-control="jump-a"]'),
          meta: item.querySelector('[data-control="meta-a"]')
        },
        b: {
          markers: [
            item.querySelector('[data-marker="b"]'),
            item.querySelector('[data-marker="b-ghost"]')
          ].filter(Boolean),
          jump: item.querySelector('[data-control="jump-b"]'),
          meta: item.querySelector('[data-control="meta-b"]')
        }
      };

      function bookmarkSummary(mark) {
        if (!Number.isFinite(mark?.seconds)) return "not set";
        const point = realTimelinePoint(videoData, mark.seconds);
        const parts = [`clip ${formatDuration(mark.seconds)}`];
        if (point) {
          parts.push(`real ${formatRealDate(point.date)}`);
          parts.push(`${point.segment.source_file || "source"}`);
        } else {
          parts.push("real unavailable");
          parts.push("file unavailable");
        }
        return parts.join(" | ");
      }

      function syncBookmarkUi() {
        if (bookmarkPanel) {
          const enabled = markerMode && hasTimeline;
          bookmarkPanel.style.display = enabled ? "grid" : "none";
          if (bookmarkNote) {
            bookmarkNote.textContent = hasTimeline
              ? "Set two tiny jump points for this long video."
              : "Markers appear only for videos with long metadata.";
          }
        }
        for (const key of ["a", "b"]) {
          const mark = bookmarks[key];
          const el = bookmarkEls[key];
          if (!el) continue;
          if (el.jump) {
            el.jump.disabled = !Number.isFinite(mark.seconds);
          }
          if (el.meta) {
            el.meta.textContent = `${key.toUpperCase()}: ${bookmarkSummary(mark)}`;
          }
          for (const markerEl of el.markers || []) {
            const duration = video.duration || 0;
            const visible = Number.isFinite(mark.seconds) && duration > 0;
            markerEl.hidden = !visible;
            if (visible) {
              const percent = Math.max(0, Math.min(100, (mark.seconds / duration) * 100));
              markerEl.style.left = `${percent}%`;
              markerEl.title = `${key.toUpperCase()} · ${bookmarkSummary(mark)}`;
            }
          }
        }
      }

      function saveAndRefresh() {
        saveBookmarks(videoData, bookmarks);
        syncBookmarkUi();
      }

      function setBookmark(key) {
        bookmarks[key].seconds = clampSeconds(video.currentTime, video.duration || 0);
        saveAndRefresh();
      }

      function jumpBookmark(key) {
        const seconds = bookmarks[key].seconds;
        if (!Number.isFinite(seconds)) return;
        video.currentTime = clampSeconds(seconds, video.duration || 0) || 0;
        update();
      }

      function resetBookmarks() {
        bookmarks.a = { seconds: null };
        bookmarks.b = { seconds: null };
        saveAndRefresh();
      }

      const activeStep = () => {
        const step = Math.trunc(Number(stepInput.value));
        const bounded = Number.isFinite(step) && step > 0 ? Math.min(step, 60) : 1;
        stepInput.value = String(bounded);
        return bounded;
      };

      const nudge = (direction) => {
        const duration = video.duration || 0;
        const next = video.currentTime + (direction * activeStep());
        video.currentTime = Math.max(0, duration ? Math.min(duration, next) : next);
        update();
      };

      const togglePlay = () => {
        if (video.paused) video.play().catch(() => {});
        else video.pause();
      };

      const update = () => {
        const duration = video.duration || 0;
        seek.value = duration ? String(Math.round((video.currentTime / duration) * 1000)) : "0";
        time.textContent = `${formatDuration(video.currentTime)} / ${formatDuration(duration)}`;
        const point = realTimelinePoint(videoData, video.currentTime);
        real.textContent = point
          ? `Real: ${formatRealDate(point.date)} · ${point.segment.source_file || "source"}`
          : "Real: unavailable";
        playBtn.textContent = video.paused ? "Play" : "Pause";
        syncBookmarkUi();
      };

      playBtn.addEventListener("click", () => {
        togglePlay();
      });
      seek.addEventListener("input", () => {
        if (video.duration) video.currentTime = (Number(seek.value) / 1000) * video.duration;
      });
      fullBtn.addEventListener("click", () => {
        if (document.fullscreenElement) document.exitFullscreen();
        else item.requestFullscreen?.();
      });
      item.querySelector('[data-control="set-a"]')?.addEventListener("click", () => setBookmark("a"));
      item.querySelector('[data-control="set-b"]')?.addEventListener("click", () => setBookmark("b"));
      item.querySelector('[data-control="reset-bookmarks"]')?.addEventListener("click", resetBookmarks);
      for (const key of ["a", "b"]) {
        bookmarkEls[key]?.jump?.addEventListener("click", () => jumpBookmark(key));
        for (const markerEl of bookmarkEls[key]?.markers || []) {
          markerEl.addEventListener("click", () => jumpBookmark(key));
        }
      }
      document.addEventListener("keydown", event => {
        const tag = document.activeElement?.tagName;
        if (tag === "INPUT" || tag === "SELECT" || tag === "TEXTAREA") return;
        if (event.code === "Space" || event.key === " ") {
          event.preventDefault();
          togglePlay();
        }
        if (event.key === "ArrowLeft") {
          event.preventDefault();
          nudge(-1);
        }
        if (event.key === "ArrowRight") {
          event.preventDefault();
          nudge(1);
        }
        if (markerMode && hasTimeline && event.key.toLowerCase() === "a") {
          event.preventDefault();
          setBookmark("a");
        }
        if (markerMode && hasTimeline && event.key.toLowerCase() === "b") {
          event.preventDefault();
          setBookmark("b");
        }
      });
      video.addEventListener("click", () => playBtn.click());
      video.addEventListener("loadedmetadata", update);
      video.addEventListener("timeupdate", update);
      video.addEventListener("play", update);
      video.addEventListener("pause", update);
      syncBookmarkUi();
      update();
    }

    function sync(method) {
      document.querySelectorAll("video").forEach(video => {
        video.playbackRate = activePlaybackRate();
        if (method === "play") video.play().catch(() => {});
        if (method === "pause") video.pause();
      });
    }

    document.getElementById("syncPlay").addEventListener("click", () => sync("play"));
    document.getElementById("syncPause").addEventListener("click", () => sync("pause"));
    timeModeEl.addEventListener("change", render);
    playbackRateEl.addEventListener("change", applyPlaybackRate);

    (async function boot() {
      try {
        if (!ids.length) {
          render();
          return;
        }
        const data = await getJson(`/api/selection?ids=${encodeURIComponent(ids.join(","))}`);
        videos = data.videos;
        render();
      } catch (err) {
        listEl.innerHTML = `<div class="empty">${escapeHtml(err.message)}</div>`;
      }
    })();
  </script>
</body>
</html>"""


def parse_args():
    parser = argparse.ArgumentParser(description="Local video review center for flip folders")
    parser.add_argument("--non-flip", type=Path, default=None, help="Optional non-flip video root")
    parser.add_argument("--flip", type=Path, default=DEFAULT_FLIP, help="Processed flip video root")
    parser.add_argument("--command-csv", type=Path, default=DEFAULT_COMMAND_CSV, help="Command export CSV/TSV")
    parser.add_argument("--remote-video-csv", type=Path, default=DEFAULT_REMOTE_VIDEO_CSV, help="Remote mp4 list CSV")
    parser.add_argument("--event-notes", type=Path, default=DEFAULT_EVENT_GROUP_NOTES, help="Saved event group names/notes JSON")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host")
    parser.add_argument("--port", type=int, default=8765, help="Bind port")
    parser.add_argument(
        "--refresh-seconds",
        type=float,
        default=5.0,
        help="Auto-refresh video index every N seconds when API requests arrive. Use 0 to disable.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    roots = {
        "flip": args.flip,
        "event": args.flip.parent / "event",
    }
    if args.non_flip is not None:
        roots["non_flip"] = args.non_flip
    index = VideoIndex(roots)
    server = ThreadingHTTPServer((args.host, args.port), ReviewHandler)
    server.index = index
    server.long_builds = LongBuildManager(args.flip, index)
    server.edge_frame_builds = EdgeFrameBuildManager(args.flip, index)
    server.event_catalog = EventCatalog(args.command_csv, args.remote_video_csv)
    server.event_group_notes = EventGroupNotes(args.event_notes)
    server.auto_refresh_seconds = max(0.0, args.refresh_seconds)

    print("Video Review Center")
    if args.non_flip is not None:
        print(f"  non-flip : {args.non_flip}")
    print(f"  flip     : {args.flip}")
    print(f"  event    : {args.flip.parent / 'event'}")
    print(f"  commands : {args.command_csv}")
    print(f"  remote   : {args.remote_video_csv}")
    print(f"  notes    : {args.event_notes}")
    print(f"  indexed  : {len(index.items)} videos")
    print(f"  sessions : {len(server.event_catalog.sessions)}")
    print(f"  refresh  : {server.auto_refresh_seconds:.1f}s")
    print(f"  open     : http://{args.host}:{args.port}/")
    print(f"  events   : http://{args.host}:{args.port}/events")
    print(f"  export   : http://{args.host}:{args.port}/events-export.html")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
