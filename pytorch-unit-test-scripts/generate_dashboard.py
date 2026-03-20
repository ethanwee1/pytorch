#!/usr/bin/env python3

import argparse
import glob
import html
import os
from collections import defaultdict
from datetime import datetime, timezone

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Generate parity dashboard HTML from CSV reports")
    parser.add_argument("--csv-dir", required=True, help="Directory containing all_tests_status CSVs")
    parser.add_argument("--output-dir", required=True, help="Output directory for dashboard HTML")
    return parser.parse_args()


def load_csvs(csv_dir):
    """Find and load all_tests_status CSVs, keyed by architecture."""
    results = {}
    for csv_path in sorted(glob.glob(os.path.join(csv_dir, "**/*_all_tests_status_*.csv"), recursive=True)):
        filename = os.path.basename(csv_path)
        for arch in ("mi200", "mi300", "mi355"):
            if f"_{arch}.csv" in filename:
                df = pd.read_csv(csv_path)
                results[arch] = {"df": df, "filename": filename, "path": csv_path}
                break
    return results


def compute_summary(df):
    """Compute high-level status counts from a parity CSV."""
    rocm_col = None
    cuda_col = None
    for col in df.columns:
        if col.startswith("status_") and cuda_col is None and rocm_col is not None:
            cuda_col = col
        if col.startswith("status_") and rocm_col is None:
            rocm_col = col

    if rocm_col is None:
        return {}

    total = len(df)
    rocm_counts = df[rocm_col].value_counts().to_dict() if rocm_col else {}
    cuda_counts = df[cuda_col].value_counts().to_dict() if cuda_col else {}

    summary = {
        "total": total,
        "rocm_col": rocm_col,
        "cuda_col": cuda_col,
        "rocm_passed": rocm_counts.get("PASSED", 0),
        "rocm_failed": rocm_counts.get("FAILED", 0) + rocm_counts.get("ERROR", 0),
        "rocm_skipped": rocm_counts.get("SKIPPED", 0),
        "rocm_missed": rocm_counts.get("MISSED", 0),
        "rocm_xfailed": rocm_counts.get("XFAILED", 0),
        "cuda_passed": cuda_counts.get("PASSED", 0),
        "cuda_failed": cuda_counts.get("FAILED", 0) + cuda_counts.get("ERROR", 0),
        "cuda_skipped": cuda_counts.get("SKIPPED", 0),
        "cuda_missed": cuda_counts.get("MISSED", 0),
    }
    return summary


def compute_skip_reasons(df):
    """Extract skip reason breakdown from a parity CSV."""
    rocm_col = None
    for col in df.columns:
        if col.startswith("status_"):
            rocm_col = col
            break

    if rocm_col is None or "skip_reason" not in df.columns:
        return {}, []

    skipped = df[df[rocm_col].isin(["SKIPPED", "MISSED"])].copy()
    if skipped.empty:
        return {}, []

    skipped["skip_reason"] = skipped["skip_reason"].fillna("Unknown / Not categorized")
    skipped["skip_reason"] = skipped["skip_reason"].replace("", "Unknown / Not categorized")

    reason_counts = skipped["skip_reason"].value_counts().to_dict()

    by_workflow = defaultdict(lambda: defaultdict(int))
    if "work_flow_name" in skipped.columns:
        for _, row in skipped.iterrows():
            wf = row.get("work_flow_name", "unknown") or "unknown"
            reason = row["skip_reason"]
            by_workflow[wf][reason] += 1

    return reason_counts, dict(by_workflow)


def compute_failure_details(df):
    """Extract failed/error tests with their messages."""
    rocm_col = None
    msg_col = None
    for col in df.columns:
        if col.startswith("status_") and rocm_col is None:
            rocm_col = col
        if col.startswith("message_") and msg_col is None:
            msg_col = col

    if rocm_col is None:
        return []

    failed = df[df[rocm_col].isin(["FAILED", "ERROR"])].copy()
    if failed.empty:
        return []

    details = []
    for _, row in failed.head(200).iterrows():
        details.append({
            "test_file": row.get("test_file", ""),
            "test_class": row.get("test_class", ""),
            "test_name": row.get("test_name", ""),
            "status": row[rocm_col],
            "workflow": row.get("work_flow_name", ""),
            "message": str(row.get(msg_col, ""))[:300] if msg_col else "",
        })
    return details


def generate_html(arch_data, output_dir):
    """Generate the full dashboard HTML."""
    os.makedirs(output_dir, exist_ok=True)

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    sha_info = ""
    for arch, data in arch_data.items():
        filename = data["filename"]
        sha_info = filename  # just use last one for display
        break

    arch_tabs_html = ""
    arch_content_html = ""

    for i, (arch, data) in enumerate(sorted(arch_data.items())):
        df = data["df"]
        summary = compute_summary(df)
        reason_counts, by_workflow = compute_skip_reasons(df)
        failures = compute_failure_details(df)

        active = "active" if i == 0 else ""

        arch_tabs_html += f'<button class="tab-btn {active}" data-tab="{arch}">{arch.upper()}</button>\n'

        # Summary cards
        cards_html = f"""
        <div class="cards">
          <div class="card card-total"><div class="card-num">{summary.get('total', 0):,}</div><div class="card-label">Total Tests</div></div>
          <div class="card card-passed"><div class="card-num">{summary.get('rocm_passed', 0):,}</div><div class="card-label">ROCm Passed</div></div>
          <div class="card card-failed"><div class="card-num">{summary.get('rocm_failed', 0):,}</div><div class="card-label">ROCm Failed/Error</div></div>
          <div class="card card-skipped"><div class="card-num">{summary.get('rocm_skipped', 0):,}</div><div class="card-label">ROCm Skipped</div></div>
          <div class="card card-missed"><div class="card-num">{summary.get('rocm_missed', 0):,}</div><div class="card-label">ROCm Missed</div></div>
          <div class="card card-cuda"><div class="card-num">{summary.get('cuda_passed', 0):,}</div><div class="card-label">CUDA Passed</div></div>
        </div>
        """

        # Skip reasons table
        skip_rows = ""
        for reason, count in sorted(reason_counts.items(), key=lambda x: -x[1]):
            escaped = html.escape(str(reason))
            skip_rows += f"<tr><td>{escaped}</td><td>{count:,}</td></tr>\n"

        skip_table_html = f"""
        <h3>Skip / Miss Reasons</h3>
        <div class="table-wrap">
          <table>
            <thead><tr><th>Skip Reason</th><th>Count</th></tr></thead>
            <tbody>{skip_rows if skip_rows else '<tr><td colspan="2">No skip reason data available</td></tr>'}</tbody>
          </table>
        </div>
        """

        # By-workflow breakdown
        wf_html = ""
        if by_workflow:
            for wf_name, reasons in sorted(by_workflow.items()):
                wf_rows = ""
                for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
                    escaped = html.escape(str(reason))
                    wf_rows += f"<tr><td>{escaped}</td><td>{count:,}</td></tr>\n"
                wf_html += f"""
                <details>
                  <summary><strong>{html.escape(wf_name)}</strong> ({sum(reasons.values()):,} skipped/missed)</summary>
                  <table>
                    <thead><tr><th>Reason</th><th>Count</th></tr></thead>
                    <tbody>{wf_rows}</tbody>
                  </table>
                </details>
                """

        # Failures table
        fail_rows = ""
        for f in failures:
            fail_rows += (
                f"<tr><td>{html.escape(str(f['test_file']))}</td>"
                f"<td>{html.escape(str(f['test_class']))}</td>"
                f"<td>{html.escape(str(f['test_name']))}</td>"
                f"<td>{html.escape(str(f['status']))}</td>"
                f"<td>{html.escape(str(f['workflow']))}</td>"
                f"<td class='msg-cell'>{html.escape(str(f['message']))}</td></tr>\n"
            )

        fail_html = ""
        if failures:
            fail_html = f"""
            <h3>Failed / Error Tests (first 200)</h3>
            <div class="table-wrap">
              <table>
                <thead><tr><th>Test File</th><th>Class</th><th>Test Name</th><th>Status</th><th>Workflow</th><th>Message</th></tr></thead>
                <tbody>{fail_rows}</tbody>
              </table>
            </div>
            """

        arch_content_html += f"""
        <div class="tab-content {active}" id="tab-{arch}">
          {cards_html}
          {skip_table_html}
          {wf_html}
          {fail_html}
        </div>
        """

    page_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>PyTorch ROCm Parity Dashboard</title>
<style>
  :root {{
    --bg: #0f1117;
    --surface: #1a1d27;
    --border: #2a2d3a;
    --text: #e1e4ed;
    --text-dim: #8b8fa3;
    --accent: #6c63ff;
    --green: #22c55e;
    --red: #ef4444;
    --yellow: #eab308;
    --blue: #3b82f6;
    --orange: #f97316;
  }}
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: var(--bg); color: var(--text); }}
  .container {{ max-width: 1400px; margin: 0 auto; padding: 24px; }}
  header {{ margin-bottom: 32px; }}
  header h1 {{ font-size: 24px; font-weight: 600; }}
  header p {{ color: var(--text-dim); margin-top: 4px; font-size: 14px; }}
  .tabs {{ display: flex; gap: 8px; margin-bottom: 24px; }}
  .tab-btn {{
    background: var(--surface); border: 1px solid var(--border); color: var(--text-dim);
    padding: 8px 20px; border-radius: 6px; cursor: pointer; font-size: 14px; font-weight: 500;
    transition: all 0.15s;
  }}
  .tab-btn:hover {{ border-color: var(--accent); color: var(--text); }}
  .tab-btn.active {{ background: var(--accent); border-color: var(--accent); color: #fff; }}
  .tab-content {{ display: none; }}
  .tab-content.active {{ display: block; }}
  .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 16px; margin-bottom: 32px; }}
  .card {{
    background: var(--surface); border: 1px solid var(--border); border-radius: 8px;
    padding: 20px; text-align: center;
  }}
  .card-num {{ font-size: 28px; font-weight: 700; }}
  .card-label {{ font-size: 13px; color: var(--text-dim); margin-top: 4px; }}
  .card-passed .card-num {{ color: var(--green); }}
  .card-failed .card-num {{ color: var(--red); }}
  .card-skipped .card-num {{ color: var(--yellow); }}
  .card-missed .card-num {{ color: var(--orange); }}
  .card-cuda .card-num {{ color: var(--blue); }}
  .card-total .card-num {{ color: var(--text); }}
  h3 {{ font-size: 18px; margin-bottom: 12px; }}
  .table-wrap {{ overflow-x: auto; margin-bottom: 24px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
  th {{ background: var(--surface); position: sticky; top: 0; text-align: left; padding: 10px 12px; border-bottom: 2px solid var(--border); }}
  td {{ padding: 8px 12px; border-bottom: 1px solid var(--border); }}
  tr:hover td {{ background: rgba(108, 99, 255, 0.05); }}
  .msg-cell {{ max-width: 400px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; font-family: monospace; font-size: 12px; color: var(--text-dim); }}
  details {{ margin-bottom: 16px; }}
  summary {{ cursor: pointer; padding: 8px 0; color: var(--accent); }}
  details table {{ margin-top: 8px; }}
</style>
</head>
<body>
<div class="container">
  <header>
    <h1>PyTorch ROCm Parity Dashboard</h1>
    <p>Generated: {now}</p>
  </header>
  <div class="tabs">
    {arch_tabs_html}
  </div>
  {arch_content_html}
</div>
<script>
document.querySelectorAll('.tab-btn').forEach(btn => {{
  btn.addEventListener('click', () => {{
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
    btn.classList.add('active');
    document.getElementById('tab-' + btn.dataset.tab).classList.add('active');
  }});
}});
</script>
</body>
</html>"""

    with open(os.path.join(output_dir, "index.html"), "w") as f:
        f.write(page_html)

    print(f"Dashboard generated at {output_dir}/index.html")
    print(f"Architectures: {', '.join(sorted(arch_data.keys()))}")
    for arch, data in sorted(arch_data.items()):
        summary = compute_summary(data["df"])
        print(f"  {arch}: {summary.get('total', 0)} tests, "
              f"{summary.get('rocm_passed', 0)} passed, "
              f"{summary.get('rocm_failed', 0)} failed, "
              f"{summary.get('rocm_skipped', 0)} skipped, "
              f"{summary.get('rocm_missed', 0)} missed")


def main():
    args = parse_args()
    arch_data = load_csvs(args.csv_dir)
    if not arch_data:
        print(f"WARNING: No all_tests_status CSVs found in {args.csv_dir}")
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, "index.html"), "w") as f:
            f.write("<html><body><h1>No data available yet</h1><p>Waiting for first daily run.</p></body></html>")
        return
    generate_html(arch_data, args.output_dir)


if __name__ == "__main__":
    main()
