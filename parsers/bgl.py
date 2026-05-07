import re
from pathlib import Path

from dataset_config import project_path


DATASET_NAME = "bgl"
HAS_LABELS = True

BGL_RE_FULL = re.compile(
    r"^(?P<label>\S+)\s+"
    r"(?P<timestamp>\d+)\s+"
    r"(?P<date>\S+)\s+"
    r"(?P<node>\S+)\s+"
    r"(?P<time>\S+)\s+"
    r"(?P<node2>\S+)\s+"
    r"(?P<type>\S+)\s+"
    r"(?P<component>\S+)\s+"
    r"(?P<level>INFO|WARN|WARNING|ERROR|FATAL|SEVERE|FAILURE|CRITICAL)"
    r"(?:\s+(?P<content>.*))?$"
)

BGL_RE_SHORT = re.compile(
    r"^(?P<label>\S+)\s+"
    r"(?P<timestamp>\d+)\s+"
    r"(?P<date>\S+)\s+"
    r"-\s+"
    r"(?P<time>\S+)\s+"
    r"(?P<type>\S+)\s+"
    r"(?P<component>\S+)\s+"
    r"(?P<level>INFO|WARN|WARNING|ERROR|FATAL|SEVERE|FAILURE|CRITICAL)"
    r"(?:\s+(?P<content>.*))?$"
)


def get_source_path() -> Path | None:
    path = project_path("data/BGL.log")
    return path if path.exists() else None


def get_source_paths() -> list[Path]:
    paths = []
    primary = get_source_path()
    if primary:
        paths.append(primary)

    data_dir = project_path("data/BGL")
    if data_dir.exists():
        paths.extend(sorted(data_dir.glob("*.log")))

    seen = set()
    unique_paths = []
    for path in paths:
        resolved = path.resolve()
        if resolved not in seen:
            unique_paths.append(path)
            seen.add(resolved)
    return unique_paths


def reset_parser_state() -> None:
    return None


def parse_line(line: str, line_id: int) -> dict | None:
    match = BGL_RE_FULL.match(line)
    fmt = "full"
    if not match:
        match = BGL_RE_SHORT.match(line)
        fmt = "short"
    if not match:
        return None

    data = match.groupdict()
    node = data.get("node", "-")
    if node == "-" or fmt == "short":
        node = "SYSTEM"

    content = (data.get("content") or "").strip() or "<EMPTY>"
    return {
        "line_id": line_id,
        "is_anomaly": int(data["label"] != "-"),
        "timestamp": int(data["timestamp"]),
        "date": data["date"],
        "node": node,
        "level": data["level"],
        "component": data["component"],
        "content": content,
    }
