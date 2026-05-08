import json
import time
import unittest
from datetime import datetime, timezone
from pathlib import Path


REPORT_DIR = Path("reports")


class ReportingResult(unittest.TextTestResult):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.successes = []

    def addSuccess(self, test):
        super().addSuccess(test)
        self.successes.append(test)


def test_id(test):
    return test.id() if hasattr(test, "id") else str(test)


def rows_for_status(status, entries, note=""):
    rows = []
    for entry in entries:
        test = entry[0] if isinstance(entry, tuple) else entry
        details = entry[1] if isinstance(entry, tuple) and status in {"FAIL", "ERROR"} else note
        if status in {"FAIL", "ERROR"}:
            details = details.strip().splitlines()[-1] if details else ""
        if status == "SKIP":
            details = entry[1]
        rows.append(
            {
                "status": status,
                "test": test_id(test),
                "notes": details,
            }
        )
    return rows


def write_reports(summary, rows):
    REPORT_DIR.mkdir(exist_ok=True)

    json_path = REPORT_DIR / "test_results.json"
    txt_path = REPORT_DIR / "test_results.txt"
    md_path = REPORT_DIR / "test_results.md"

    json_path.write_text(
        json.dumps({"summary": summary, "results": rows}, indent=2),
        encoding="utf-8",
    )

    txt_lines = [
        "AIOps Test Results",
        "=" * 18,
        f"Generated at: {summary['generated_at']}",
        f"Total tests: {summary['total']}",
        f"Passed: {summary['passed']}",
        f"Failed: {summary['failed']}",
        f"Errors: {summary['errors']}",
        f"Skipped: {summary['skipped']}",
        f"Duration seconds: {summary['duration_seconds']}",
        "",
        "Detailed results:",
    ]
    txt_lines.extend(f"[{row['status']}] {row['test']} {row['notes']}".rstrip() for row in rows)
    txt_path.write_text("\n".join(txt_lines) + "\n", encoding="utf-8")

    md_lines = [
        "# AIOps Test Results",
        "",
        f"- Generated at: `{summary['generated_at']}`",
        f"- Total tests: `{summary['total']}`",
        f"- Passed: `{summary['passed']}`",
        f"- Failed: `{summary['failed']}`",
        f"- Errors: `{summary['errors']}`",
        f"- Skipped: `{summary['skipped']}`",
        f"- Duration seconds: `{summary['duration_seconds']}`",
        "",
        "| Status | Test Case | Notes |",
        "|---|---|---|",
    ]
    for row in rows:
        notes = row["notes"].replace("|", "\\|")
        md_lines.append(f"| {row['status']} | `{row['test']}` | {notes} |")
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    return json_path, txt_path, md_path


def main():
    start = time.perf_counter()
    suite = unittest.defaultTestLoader.discover("tests")
    runner = unittest.TextTestRunner(verbosity=2, resultclass=ReportingResult)
    result = runner.run(suite)
    duration = round(time.perf_counter() - start, 3)

    rows = []
    rows.extend(rows_for_status("PASS", result.successes))
    rows.extend(rows_for_status("FAIL", result.failures))
    rows.extend(rows_for_status("ERROR", result.errors))
    rows.extend(rows_for_status("SKIP", result.skipped))
    rows.sort(key=lambda row: row["test"])

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "total": result.testsRun,
        "passed": len(result.successes),
        "failed": len(result.failures),
        "errors": len(result.errors),
        "skipped": len(result.skipped),
        "duration_seconds": duration,
        "successful": result.wasSuccessful(),
    }

    paths = write_reports(summary, rows)
    print()
    print("Test reports written:")
    for path in paths:
        print(f"  {path}")

    raise SystemExit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__":
    main()
