import unittest

from notify_parity_failures import GhError, Reporter

ISSUE_REPOSITORY = "ethanwee1/pytorch"
TARGET_REPOSITORY = "ROCm/pytorch"
CREATED_AT = "2026-10-29T20:00:00Z"
RUN_URL = "https://github.com/ROCm/pytorch/actions/runs/123"


def failed_run(url=RUN_URL):
    return {
        "name": "Parity Report",
        "conclusion": "failure",
        "created_at": "2026-10-29T20:05:00Z",
        "html_url": url,
        "head_sha": "a" * 40,
    }


class FakeGhClient:
    def __init__(self, tracker=None, comments=None, runs=None):
        self.issues = [tracker] if tracker else []
        self.issue_comments = comments or {}
        self.runs = runs or {}
        self.created = []
        self.closed = []
        self.posted = []
        self.failed_run_calls = []
        self.comment_errors = []

    def open_issues(self, issue_repository):
        return self.issues

    def comments(self, issue_repository, issue_number):
        return self.issue_comments.get(issue_number, [])

    def create_issue(self, issue_repository, title, body):
        issue = {
            "number": 9000 + len(self.created),
            "created_at": CREATED_AT,
            "title": title,
            "body": body,
        }
        self.created.append(issue)
        self.issues.append(issue)
        return issue

    def close_issue(self, issue_repository, issue_number):
        self.closed.append(issue_number)

    def comment_issue(self, issue_repository, issue_number, body):
        if self.comment_errors:
            raise self.comment_errors.pop(0)
        self.posted.append((issue_number, body))

    def failed_runs(self, target_repository, workflow, since):
        self.failed_run_calls.append((workflow, since))
        return self.runs.get(workflow, [])


def tracker(number=10868):
    return {
        "number": number,
        "created_at": "2026-10-01T00:00:00Z",
        "body": f"<!-- parity-failure-reporter:{TARGET_REPOSITORY} -->",
    }


def reporter(client, **overrides):
    options = {
        "issue_repository": ISSUE_REPOSITORY,
        "target_repository": TARGET_REPOSITORY,
        "dry_run": False,
        "monitor_since": None,
        "rollover_threshold": 2400,
    }
    options.update(overrides)
    return Reporter(client, **options)


class ReporterTest(unittest.TestCase):
    def test_existing_run_url_is_not_reported_twice(self):
        existing_comment = {
            "body": f"Failure details\nRun: {RUN_URL}\nCommit: {'a' * 40}"
        }
        client = FakeGhClient(
            tracker=tracker(),
            comments={10868: [existing_comment]},
            runs={"parity.yml": [failed_run()]},
        )

        result = reporter(client).run()

        self.assertEqual(result, {"posted": 0, "skipped": 1})
        self.assertEqual(client.posted, [])

    def test_rollover_preserves_unreported_runs(self):
        client = FakeGhClient(
            tracker=tracker(),
            comments={10868: [{"body": "reported"}] * 2400},
            runs={"parity.yml": [failed_run()]},
        )

        result = reporter(client).run()

        self.assertEqual(result, {"posted": 1, "skipped": 0})
        self.assertEqual(client.closed, [10868])
        self.assertEqual(client.posted[0][0], 9000)
        self.assertEqual(client.failed_run_calls[0][1], "2026-10-01T00:00:00Z")

    def test_new_tracker_uses_its_creation_time_as_cutoff(self):
        client = FakeGhClient()

        reporter(client).run()

        self.assertEqual(len(client.created), 1)
        self.assertEqual(
            {since for _, since in client.failed_run_calls},
            {CREATED_AT},
        )

    def test_comment_cap_error_rolls_over_and_retries_once(self):
        client = FakeGhClient(
            tracker=tracker(),
            runs={"parity.yml": [failed_run()]},
        )
        client.comment_errors.append(
            GhError("Commenting is disabled on issues with more than 2500 comments")
        )

        result = reporter(client).run()

        self.assertEqual(result, {"posted": 1, "skipped": 0})
        self.assertEqual(client.closed, [10868])
        self.assertEqual(client.posted[0][0], 9000)

    def test_unrelated_comment_error_is_not_hidden(self):
        client = FakeGhClient(
            tracker=tracker(),
            runs={"parity.yml": [failed_run()]},
        )
        client.comment_errors.append(GhError("HTTP 403: forbidden"))

        with self.assertRaisesRegex(GhError, "403"):
            reporter(client).run()

        self.assertEqual(client.created, [])
        self.assertEqual(client.closed, [])


if __name__ == "__main__":
    unittest.main()
