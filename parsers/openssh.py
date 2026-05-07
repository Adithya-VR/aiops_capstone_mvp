import re
from datetime import datetime
from pathlib import Path

from dataset_config import project_path


DATASET_NAME = "openssh"
WINDOW_SECS = 300
STEP_SECS = 150
HAS_LABELS = False
BASE_YEAR = 2004
_current_year = BASE_YEAR
_previous_month = None

SSH_RE = re.compile(
    r"^(?P<month>\w+)\s+(?P<day>\d+)\s+(?P<time>\d+:\d+:\d+)\s+"
    r"(?P<host>\S+)\s+sshd\[(?P<pid>\d+)\]:\s+(?P<content>.+)$"
)

IP_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
USER_PATTERNS = [
    re.compile(r"Invalid user (?P<user>\S+) from", re.IGNORECASE),
    re.compile(r"Failed password for invalid user (?P<user>\S+) from", re.IGNORECASE),
    re.compile(r"Failed password for (?P<user>\S+) from", re.IGNORECASE),
    re.compile(r"user=(?P<user>\S+)", re.IGNORECASE),
]

MONTH_MAP = {
    "Jan": 1,
    "Feb": 2,
    "Mar": 3,
    "Apr": 4,
    "May": 5,
    "Jun": 6,
    "Jul": 7,
    "Aug": 8,
    "Sep": 9,
    "Oct": 10,
    "Nov": 11,
    "Dec": 12,
}


def get_source_path() -> Path | None:
    paths = get_source_paths()
    return paths[0] if paths else None


def get_source_paths() -> list[Path]:
    data_dir = project_path("data/OpenSSH")
    if not data_dir.exists():
        return []

    return sorted(data_dir.glob("*.log"))


def reset_parser_state() -> None:
    global _current_year, _previous_month
    _current_year = BASE_YEAR
    _previous_month = None


def _parse_time(month: str, day: str, time_str: str) -> int:
    global _current_year, _previous_month

    try:
        month_num = MONTH_MAP[month]
        if _previous_month is not None and month_num < _previous_month:
            _current_year += 1
        _previous_month = month_num

        hour, minute, second = map(int, time_str.split(":"))
        dt = datetime(
            _current_year,
            month_num,
            int(day),
            hour,
            minute,
            second,
        )
        return int(dt.timestamp())
    except Exception:
        return None


def _is_anomaly(_content: str) -> int:
    # The raw OpenSSH source has no ground-truth labels. Attack-like strings
    # are used as features, not as supervised truth.
    return 0


def _get_level(content: str) -> str:
    c = content.lower()
    if "possible break-in attempt" in c:
        return "BREAKIN_ATTEMPT"
    if "invalid user" in c:
        return "INVALID_USER"
    if "failed password" in c:
        return "FAILED_PASSWORD"
    if "authentication failure" in c:
        return "AUTH_FAILURE"
    if "too many authentication failures" in c or "maximum authentication" in c:
        return "TOO_MANY_FAILURES"
    if "bad protocol version" in c or "did not receive" in c:
        return "BAD_PROTOCOL"
    if "accepted password" in c or "accepted publickey" in c:
        return "ACCEPTED_AUTH"
    if "received disconnect" in c or "disconnecting:" in c:
        return "DISCONNECT"
    if "connection closed" in c:
        return "CONNECTION_CLOSED"
    if "pam_unix" in c or "pam service" in c:
        return "PAM"
    if "fatal:" in c:
        return "FATAL"
    if "error:" in c:
        return "ERROR"
    if "warning" in c:
        return "WARNING"
    return "INFO"


def _extract_source_ip(content: str) -> str:
    match = IP_RE.search(content)
    return match.group(0) if match else ""


def _extract_user(content: str) -> str:
    for pattern in USER_PATTERNS:
        match = pattern.search(content)
        if match:
            return match.group("user").strip()
    return ""


def parse_line(line: str, line_id: int) -> dict | None:
    match = SSH_RE.match(line)
    if not match:
        return None

    data = match.groupdict()
    content = data["content"]
    return {
        "line_id": line_id,
        "is_anomaly": _is_anomaly(content),
        "timestamp": _parse_time(data["month"], data["day"], data["time"]),
        "date": f"{data['month']} {data['day']}",
        "node": data["host"],
        "level": _get_level(content),
        "component": "sshd",
        "content": content,
        "source_ip": _extract_source_ip(content),
        "user": _extract_user(content),
    }
