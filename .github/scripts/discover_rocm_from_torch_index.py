#!/usr/bin/env python3
"""
discover_rocm_from_torch_index.py

Parse a ROCm PyTorch wheel index HTML, pick the latest torch wheel matching an
optional PEP 440-style prefix (from release/x.y branches), and emit step outputs
via GITHUB_OUTPUT (or legacy ::set-output when that variable is unset).

Usage (from repo root, as in GitHub Actions):

    python3 .github/scripts/discover_rocm_from_torch_index.py \\
        --index-url <BASE_URL> \\
        --amdgpu-family <FAMILY> \\
        [--torch-version-prefix <PREFIX>]
"""

from __future__ import annotations

import argparse
import os
import re
import urllib.parse
import urllib.request
from typing import Any


def _version_sort_key(v: str) -> tuple[int, ...]:
    try:
        return tuple(int(x) for x in re.split(r"[.\-a+]", v) if x.isdigit())
    except (ValueError, AttributeError):
        return (0,)


def discover_rocm_version(
    index_url: str,
    gpu_family: str,
    torch_version_prefix: str,
    *,
    timeout_s: int = 60,
) -> tuple[str, str]:
    """Return (rocm_version, latest_torch_wheel_version_string)."""
    url = f"{index_url.rstrip('/')}/{gpu_family}/torch/"
    print(f"Fetching torch index: {url}")
    html = urllib.request.urlopen(url, timeout=timeout_s).read().decode()

    pattern = re.compile(r"torch-(.+?)\.whl", re.IGNORECASE)
    versions: list[str] = []
    for m in pattern.finditer(html):
        ver = urllib.parse.unquote(m.group(1).split("-")[0])
        if "+rocm" in ver:
            versions.append(ver)

    if torch_version_prefix:
        versions = [v for v in versions if v.split("+")[0].startswith(torch_version_prefix)]

    if not versions:
        print(f"::error::No torch wheels found (prefix={torch_version_prefix!r})")
        raise SystemExit(1)

    latest = max(versions, key=_version_sort_key)
    match = re.search(r"\+rocm(.+)", latest)
    if not match:
        print(f"::error::Could not parse ROCm suffix from wheel version {latest!r}")
        raise SystemExit(1)
    rocm_ver = match.group(1)

    print(f"Latest torch wheel: {latest}")
    print(f"Discovered ROCm version: {rocm_ver}")
    return rocm_ver, latest


def set_output(name: str, val: Any) -> None:
    print(f"Setting output {name}={val}")
    if os.getenv("GITHUB_OUTPUT"):
        with open(str(os.getenv("GITHUB_OUTPUT")), "a") as env:
            print(f"{name}={val}", file=env)
    else:
        print(f"::set-output name={name}::{val}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Discover ROCm version from a PyTorch ROCm wheel index page.",
    )
    parser.add_argument(
        "--index-url",
        required=True,
        help="Base index URL (e.g. https://rocm.nightlies.amd.com/v2-staging)",
    )
    parser.add_argument(
        "--amdgpu-family",
        required=True,
        help="GPU family subdirectory under the index (e.g. gfx94X-dcgpu)",
    )
    parser.add_argument(
        "--torch-version-prefix",
        default="",
        help="If set, only wheels whose version starts with this prefix (e.g. 2.11)",
    )
    args = parser.parse_args()

    rocm_ver, latest = discover_rocm_version(
        args.index_url,
        args.amdgpu_family,
        args.torch_version_prefix,
    )
    set_output("rocm_version", rocm_ver)
    set_output("torch_wheel_version", latest)


if __name__ == "__main__":
    main()
