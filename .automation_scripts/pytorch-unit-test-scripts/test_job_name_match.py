import unittest

from job_name_match import choose_fuzzy_job_prefix


class JobNameMatchTest(unittest.TestCase):
    def test_fuzzy_match_selects_renamed_prefix_for_requested_arch(self):
        configured = "linux-jammy-rocm-py3.10-mi300"
        renamed = "linux-noble-rocm-py3.12-mi300"
        names = [
            f"{renamed} / test (default, 1, 8, rocm.gpu)",
            f"{renamed} / test (default, 2, 8, rocm.gpu)",
            "linux-noble-rocm-py3.12-mi350 / test "
            "(default, 1, 8, rocm.gpu)",
            "linux-jammy-cuda13.0-py3.10-gcc11 / test "
            "(default, 1, 14, nvidia.gpu)",
        ]

        self.assertEqual(
            choose_fuzzy_job_prefix(
                names, "default", configured, arch="mi300"
            ),
            renamed,
        )

    def test_exact_match_wins(self):
        configured = "linux-noble-rocm-py3.12-mi300"
        names = [
            f"{configured} / test (distributed, 1, 5, rocm.gpu)",
            "linux-noble-rocm-py3.13-mi300 / test "
            "(distributed, 1, 5, rocm.gpu)",
        ]

        self.assertEqual(
            choose_fuzzy_job_prefix(
                names, "distributed", configured, arch="mi300"
            ),
            configured,
        )

    def test_preview_does_not_select_non_preview_mi350(self):
        configured = "linux-noble-rocm-preview-py3.12-mi350"
        names = [
            "linux-noble-rocm-py3.12-mi350 / test "
            "(inductor, 1, 2, rocm.gpu)",
        ]

        self.assertEqual(
            choose_fuzzy_job_prefix(
                names, "inductor", configured, arch="preview"
            ),
            configured,
        )

    def test_no_matching_config_keeps_configured_prefix(self):
        configured = "linux-jammy-rocm-py3.10-mi200"
        names = [
            "linux-jammy-rocm-py3.10-mi200 / test "
            "(default, 1, 10, rocm.gpu)",
        ]

        self.assertEqual(
            choose_fuzzy_job_prefix(
                names, "distributed", configured, arch="mi200"
            ),
            configured,
        )


if __name__ == "__main__":
    unittest.main()
