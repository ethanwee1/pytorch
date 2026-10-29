"""Report failed parity runs once and rotate exhausted tracker issues."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import UTC, datetime
from typing import Any
from urllib.parse import quote, urlencode

FAILURE_CONCLUSIONS = {"failure", "timed_out", "startup_failure"}
WORKFLOWS = ("parity.yml", "parity-auto.yml")
DEFAULT_ISSUE_REPOSITORY = "ethanwee1/pytorch"
DEFAULT_ROLLOVER_THRESHOLD = 2400
RUN_URL_RE = re.compile(
    r"^Run: (https://github\.com/[^\s]+/actions/runs/\d+)$", re.MULTILINE
)
REPOSITORY_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")


class GhError(RuntimeError):
    """A GitHub CLI request failed."""


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def tracker_title(target_repository: str) -> str:
    return f"Parity workflow failures: {target_repository}"


def tracker_marker(target_repository: str) -> str:
    return f"<!-- parity-failure-reporter:{target_repository} -->"


def tracker_body(target_repository: str) -> str:
    return (
        f"Tracks failed parity workflow runs in `{target_repository}`.\n\n"
        f"{tracker_marker(target_repository)}"
    )


def extract_reported_run_urls(comments: list[dict[str, Any]]) -> set[str]:
    return {
        match.group(1)
        for comment in comments
        for match in RUN_URL_RE.finditer(comment.get("body") or "")
    }


def is_comment_capacity_error(error: GhError) -> bool:
    message = str(error).lower()
    return "commenting is disabled" in message and "2500 comments" in message


class GhClient:
    def __init__(self, token: str | None = None) -> None:
        self.env = os.environ.copy()
        token = token or self.env.get("GH_TOKEN") or self.env.get("GITHUB_TOKEN")
        if token:
            self.env["GH_TOKEN"] = token

    def api(
        self,
        endpoint: str,
        *,
        method: str = "GET",
        fields: dict[str, str] | None = None,
    ) -> Any:
        command = ["gh", "api", endpoint]
        if method != "GET":
            command.extend(["--method", method])
        for key, value in (fields or {}).items():
            command.extend(["-f", f"{key}={value}"])
        proc = subprocess.run(
            command,
            capture_output=True,
            check=False,
            env=self.env,
            text=True,
        )
        if proc.returncode != 0:
            raise GhError((proc.stderr or proc.stdout or "gh api failed").strip())
        return json.loads(proc.stdout) if proc.stdout.strip() else None

    def paginated(
        self,
        path: str,
        params: dict[str, str],
        *,
        item_key: str | None = None,
    ) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        page = 1
        while True:
            query = urlencode({**params, "per_page": "100", "page": str(page)})
            response = self.api(f"{path}?{query}")
            batch = response[item_key] if item_key else response
            items.extend(batch)
            if len(batch) < 100:
                return items
            page += 1

    def open_issues(self, issue_repository: str) -> list[dict[str, Any]]:
        return self.paginated(
            f"repos/{issue_repository}/issues",
            {"state": "open"},
        )

    def comments(
        self, issue_repository: str, issue_number: int
    ) -> list[dict[str, Any]]:
        return self.paginated(
            f"repos/{issue_repository}/issues/{issue_number}/comments",
            {},
        )

    def create_issue(
        self,
        issue_repository: str,
        title: str,
        body: str,
    ) -> dict[str, Any]:
        return self.api(
            f"repos/{issue_repository}/issues",
            method="POST",
            fields={"title": title, "body": body},
        )

    def close_issue(self, issue_repository: str, issue_number: int) -> None:
        self.api(
            f"repos/{issue_repository}/issues/{issue_number}",
            method="PATCH",
            fields={"state": "closed"},
        )

    def comment_issue(
        self, issue_repository: str, issue_number: int, body: str
    ) -> None:
        self.api(
            f"repos/{issue_repository}/issues/{issue_number}/comments",
            method="POST",
            fields={"body": body},
        )

    def failed_runs(
        self,
        target_repository: str,
        workflow: str,
        since: str,
    ) -> list[dict[str, Any]]:
        runs = self.paginated(
            f"repos/{target_repository}/actions/workflows/{quote(workflow, safe='')}/runs",
            {"status": "completed", "created": f">={since}"},
            item_key="workflow_runs",
        )
        return [run for run in runs if run.get("conclusion") in FAILURE_CONCLUSIONS]


class Reporter:
    def __init__(
        self,
        client: GhClient,
        *,
        issue_repository: str,
        target_repository: str,
        dry_run: bool,
        monitor_since: str | None,
        rollover_threshold: int,
    ) -> None:
        self.client = client
        self.issue_repository = issue_repository
        self.target_repository = target_repository
        self.dry_run = dry_run
        self.monitor_since = monitor_since
        self.rollover_threshold = rollover_threshold

    def find_tracker(self) -> dict[str, Any] | None:
        marker = tracker_marker(self.target_repository)
        candidates = [
            issue
            for issue in self.client.open_issues(self.issue_repository)
            if "pull_request" not in issue and marker in (issue.get("body") or "")
        ]
        return (
            max(candidates, key=lambda issue: issue.get("created_at", ""))
            if candidates
            else None
        )

    def create_tracker(self, created_at: str) -> dict[str, Any]:
        if self.dry_run:
            print(f"Would create tracker: {tracker_title(self.target_repository)}")
            return {"number": 0, "created_at": created_at, "comments": 0}
        issue = self.client.create_issue(
            self.issue_repository,
            tracker_title(self.target_repository),
            tracker_body(self.target_repository),
        )
        print(f"Created tracker #{issue['number']} for {self.target_repository}")
        return issue

    def rollover(self, current: dict[str, Any], created_at: str) -> dict[str, Any]:
        replacement = self.create_tracker(created_at)
        if not self.dry_run:
            self.client.close_issue(self.issue_repository, current["number"])
            print(f"Closed exhausted tracker #{current['number']}")
        else:
            print(f"Would close exhausted tracker #{current['number']}")
        return replacement

    def format_comment(self, run: dict[str, Any]) -> str:
        return "\n".join(
            (
                (
                    f"{run['name']} failed in {self.target_repository} "
                    f"({run['conclusion']}, {run['created_at']})."
                ),
                f"Run: {run['html_url']}",
                f"Commit: {run['head_sha']}",
            )
        )

    def run(self) -> dict[str, int]:
        now = utc_now()
        tracker = self.find_tracker()
        if tracker is None:
            tracker = self.create_tracker(now)
            comments: list[dict[str, Any]] = []
            scan_since = self.monitor_since or tracker["created_at"]
        else:
            comments = self.client.comments(self.issue_repository, tracker["number"])
            scan_since = self.monitor_since or tracker["created_at"]

        reported = extract_reported_run_urls(comments)
        if len(comments) >= self.rollover_threshold:
            tracker = self.rollover(tracker, now)

        runs = [
            run
            for workflow in WORKFLOWS
            for run in self.client.failed_runs(
                self.target_repository,
                workflow,
                scan_since,
            )
        ]
        runs.sort(key=lambda run: (run["created_at"], run["html_url"]))

        posted = 0
        skipped = 0
        for run in runs:
            run_url = run["html_url"]
            if run_url in reported:
                skipped += 1
                continue
            body = self.format_comment(run)
            if self.dry_run:
                print(f"Would notify tracker #{tracker['number']}:\n{body}")
            else:
                try:
                    self.client.comment_issue(
                        self.issue_repository,
                        tracker["number"],
                        body,
                    )
                except GhError as error:
                    if not is_comment_capacity_error(error):
                        raise
                    tracker = self.rollover(tracker, now)
                    self.client.comment_issue(
                        self.issue_repository,
                        tracker["number"],
                        body,
                    )
            reported.add(run_url)
            posted += 1

        print(
            f"{self.target_repository}: {posted} notification(s), "
            f"{skipped} already reported"
        )
        return {"posted": posted, "skipped": skipped}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-repository", required=True)
    parser.add_argument(
        "--issue-repository",
        default=DEFAULT_ISSUE_REPOSITORY,
    )
    parser.add_argument("--monitor-since")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--rollover-threshold",
        type=int,
        default=DEFAULT_ROLLOVER_THRESHOLD,
    )
    return parser.parse_args()


def main() -> int:
    args = parse_arguments()
    for repository in (args.target_repository, args.issue_repository):
        if not REPOSITORY_RE.fullmatch(repository):
            print(f"error: invalid repository {repository!r}", file=sys.stderr)
            return 2
    if args.rollover_threshold < 1 or args.rollover_threshold > 2500:
        print("error: rollover threshold must be between 1 and 2500", file=sys.stderr)
        return 2

    Reporter(
        GhClient(),
        issue_repository=args.issue_repository,
        target_repository=args.target_repository,
        dry_run=args.dry_run,
        monitor_since=args.monitor_since,
        rollover_threshold=args.rollover_threshold,
    ).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
