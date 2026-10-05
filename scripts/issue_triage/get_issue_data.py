#!/usr/bin/env python3
"""Fetch all GitHub issues for a repo and dump a JSONL file.

Uses ``gh api`` (already authenticated) to avoid managing tokens.
Output fields per line: id, title, author, created_at, open, body.
"""

import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent
OUTPUT = HERE / "issues.jsonl"
REPO = "mflux-community/mflux"
PER_PAGE = 100


def run_gh(endpoint: str) -> list[dict]:
    """Call ``gh api`` and return parsed JSON (list of objects)."""
    result = subprocess.run(
        ["gh", "api", endpoint],
        capture_output=True,
        text=True,
        timeout=60,
    )
    if result.returncode != 0:
        print(f"gh api failed: {endpoint}\n{result.stderr}", file=sys.stderr)
        sys.exit(1)
    return json.loads(result.stdout)


def fetch_all_issues() -> list[dict]:
    """Paginate through all issues (open + closed)."""
    all_issues: list[dict] = []
    page = 1
    while True:
        endpoint = (
            f"repos/{REPO}/issues?state=all&per_page={PER_PAGE}&page={page}&sort=created&direction=asc&filter=all"
        )
        page_issues = run_gh(endpoint)
        # Exclude pull requests (they share the issues endpoint)
        issues = [i for i in page_issues if "pull_request" not in i]
        all_issues.extend(issues)
        print(f"Page {page}: {len(issues)} issues (total {len(all_issues)})")
        if len(page_issues) < PER_PAGE:
            break
        page += 1
    return all_issues


def main() -> int:
    issues = fetch_all_issues()
    print(f"\nTotal issues: {len(issues)}")

    with open(OUTPUT, "w") as f:
        for issue in issues:
            record = {
                "id": issue["number"],
                "title": issue["title"],
                "author": issue["user"]["login"],
                "created_at": issue["created_at"],
                "open": issue["state"] == "open",
                "body": issue.get("body") or "",
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Wrote {OUTPUT} ({OUTPUT.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
