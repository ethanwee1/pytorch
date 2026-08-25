import difflib
import re


_ARCH_PREFIX_TOKENS = {
    "mi200": ("mi200", "mi210"),
    "mi300": ("mi300",),
    "mi350": ("mi350",),
    "navi31": ("navi31",),
    "preview": ("rocm-preview",),
}


def _is_rocm_job_prefix(prefix):
    prefix_lower = prefix.lower()
    return "rocm" in prefix_lower and "cuda" not in prefix_lower


def _prefix_matches_arch(arch, prefix, configured_prefix):
    prefix_lower = prefix.lower()
    tokens = _ARCH_PREFIX_TOKENS.get(arch, ())
    if any(token in prefix_lower for token in tokens):
        return True
    if prefix == configured_prefix:
        return True
    configured_lower = configured_prefix.lower()
    if not any(token in configured_lower for token in tokens):
        return _is_rocm_job_prefix(prefix)
    return False


def _candidate_job_prefixes(names, test_config):
    """Count prefixes from '<prefix> / <kind> (<config>, <idx>, <total>, ...)'."""
    pattern = re.compile(
        r"^(.+?) / \S+ \(" + re.escape(test_config) + r", \d+, \d+"
    )
    candidates = {}
    for name in names:
        match = pattern.match(name)
        if match:
            prefix = match.group(1)
            candidates[prefix] = candidates.get(prefix, 0) + 1
    return candidates


def choose_fuzzy_job_prefix(names, test_config, configured_prefix, arch=None):
    """Choose the closest eligible ROCm prefix, preserving an exact match."""
    candidates = {
        prefix: count
        for prefix, count in _candidate_job_prefixes(names, test_config).items()
        if _is_rocm_job_prefix(prefix)
        and (
            arch is None
            or _prefix_matches_arch(arch, prefix, configured_prefix)
        )
    }
    if not candidates or configured_prefix in candidates:
        return configured_prefix
    return max(
        candidates,
        key=lambda prefix: (
            difflib.SequenceMatcher(
                None, prefix, configured_prefix
            ).ratio(),
            candidates[prefix],
        ),
    )
