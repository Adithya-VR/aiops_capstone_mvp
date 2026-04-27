# run_this.py - find lines where level is garbage
import re
import pandas as pd

BGL_RE = re.compile(
    r'^(?P<label>\S+)\s+'
    r'(?P<timestamp>\d+)\s+'
    r'(?P<date>\S+)\s+'
    r'(?P<node>\S+)\s+'
    r'(?P<time>\S+)\s+'
    r'(?P<node2>\S+)\s+'
    r'(?P<type>\S+)\s+'
    r'(?P<component>\S+)\s+'
    r'(?P<level>\S+)\s+'
    r'(?P<content>.+)$'
)

VALID_LEVELS = {
    "INFO", "WARN", "WARNING", "ERROR",
    "FATAL", "SEVERE", "FAILURE", "CRITICAL"
}

bad_lines = []
with open("data/BGL.log", encoding="utf-8", errors="replace") as f:
    for i, line in enumerate(f):
        m = BGL_RE.match(line.strip())
        if m:
            d = m.groupdict()
            if d["level"] not in VALID_LEVELS:
                bad_lines.append({
                    "line_num": i,
                    "raw":      line.strip()[:200],
                    "captured_level": d["level"],
                    "captured_content": d["content"][:80]
                })
        if i > 4500000:  # check first 500K lines
            break
        if len(bad_lines) >= 20:
            break

print(f"Found {len(bad_lines)} bad lines in first 500K")
for b in bad_lines[:10]:
    print(f"\nLine {b['line_num']}:")
    print(f"  Raw     : {b['raw']}")
    print(f"  Level   : {b['captured_level']}")
    print(f"  Content : {b['captured_content']}")