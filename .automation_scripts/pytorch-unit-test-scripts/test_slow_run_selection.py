import importlib.machinery
import importlib.util
import os
import sys
import unittest


sys.path.insert(0, os.path.dirname(__file__))

# download_testlogs is a CLI without a .py suffix and refuses to import without
# credentials, so load it by path with placeholders.
for _var in ("GITHUB_TOKEN", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"):
    os.environ.setdefault(_var, "placeholder")

_PATH = os.path.join(os.path.dirname(__file__), "download_testlogs")
_spec = importlib.util.spec_from_loader(
    "download_testlogs", importlib.machinery.SourceFileLoader("download_testlogs", _PATH)
)
dtl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dtl)


# slow.yml runs for fc6c9cdfd4c55eb9a9df3f3923b8d808c1cf1ae3: the push run holds
# the real shards, the scheduled run re-ran only the rerun_disabled_tests
# variants a minute later and so comes back first from the API.
PUSH_RUN = 35976419544
VARIANT_RUN = 35976562424
ROCM_PREFIX = "linux-noble-rocm-py3.11-mi350"
CUDA_PREFIX = "linux-jammy-cuda13.0-py3.10-gcc11-sm86"


def _jobs(run_id, prefix):
    suffix = ", rerun_disabled_tests" if run_id == VARIANT_RUN else ""
    base = 10756412000 if run_id == VARIANT_RUN else 10756383000
    return [
        {"id": base + i, "name": f"{prefix} / test (slow, {i}, 3, runner{suffix})"}
        for i in (1, 2, 3)
    ]


class SlowRunSelectionTest(unittest.TestCase):
    def setUp(self):
        self.runs = [
            {"id": VARIANT_RUN, "event": "schedule", "status": "completed"},
            {"id": PUSH_RUN, "event": "push", "status": "completed"},
        ]
        self.prefix = ROCM_PREFIX
        self._orig = (dtl.requests.get, dtl.get_workflow_jobs)

        class Resp:
            def json(_self):
                return {"workflow_runs": self.runs}

        dtl.requests.get = lambda *a, **k: Resp()
        dtl.get_workflow_jobs = lambda run, all_attempts=False: _jobs(run["id"], self.prefix)

    def tearDown(self):
        dtl.requests.get, dtl.get_workflow_jobs = self._orig

    def test_picks_push_run_over_variant_rerun(self):
        for prefix in (ROCM_PREFIX, CUDA_PREFIX):
            self.prefix = prefix
            run = dtl.resolve_non_variant_run("slow", "fc6c9cd", "slow", prefix)
            self.assertEqual(run["id"], PUSH_RUN)

    def test_declines_when_only_variant_shards_exist(self):
        self.runs = [{"id": VARIANT_RUN, "event": "schedule", "status": "completed"}]
        self.assertIsNone(
            dtl.resolve_non_variant_run("slow", "fc6c9cd", "slow", ROCM_PREFIX)
        )


class ArtifactRedirectTest(unittest.TestCase):
    def setUp(self):
        check_runs = []
        for run_id in (PUSH_RUN, VARIANT_RUN):
            for job in _jobs(run_id, ROCM_PREFIX):
                check_runs.append({
                    "name": job["name"],
                    "details_url": (
                        f"https://github.com/pytorch/pytorch/actions/runs/{run_id}"
                        f"/job/{job['id']}"
                    ),
                })
        self._orig = (dtl.get_check_runs_for_commit, dtl.get_run_by_id)
        dtl.get_check_runs_for_commit = lambda sha, prefix: check_runs
        dtl.get_run_by_id = lambda rid: {"id": int(rid), "name": "slow"}

    def tearDown(self):
        dtl.get_check_runs_for_commit, dtl.get_run_by_id = self._orig

    def test_stays_on_push_run(self):
        wf, substrings = dtl.resolve_artifact_download(
            {"id": PUSH_RUN, "head_sha": "fc6c9cd"}, ROCM_PREFIX, "slow", ["rocm.gpu"]
        )
        self.assertEqual(wf["id"], PUSH_RUN)
        self.assertEqual(substrings, ["rocm.gpu"])

    def test_redirects_to_push_run_never_the_variant_run(self):
        wf, substrings = dtl.resolve_artifact_download(
            {"id": 999, "head_sha": "fc6c9cd"}, ROCM_PREFIX, "slow", ["rocm.gpu"]
        )
        self.assertEqual(wf["id"], PUSH_RUN)
        self.assertEqual(substrings, ["_10756383001", "_10756383002", "_10756383003"])


if __name__ == "__main__":
    unittest.main()
