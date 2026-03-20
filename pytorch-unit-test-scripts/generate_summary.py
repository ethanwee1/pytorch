#!/usr/bin/env python3

import argparse
import csv
import sys


WORKFLOWS = ['default', 'distributed', 'inductor']
WORKFLOW_DISPLAY = {
    'default': 'TEST DEFAULT',
    'distributed': 'TEST DISTRIBUTED',
    'inductor': 'TEST INDUCTOR',
}


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate a parity summary CSV from per-architecture test status CSVs'
    )
    parser.add_argument(
        '--csv', nargs='+', required=True,
        help='CSV file(s) to summarize (one per architecture, same order as --arch)'
    )
    parser.add_argument(
        '--arch', nargs='+', required=True,
        help='Architecture labels matching --csv order (e.g. mi200 mi300 mi355)'
    )
    parser.add_argument('--sha', type=str, default='', help='Commit SHA')
    parser.add_argument('--pr_id', type=str, default='', help='Pull request ID')
    parser.add_argument(
        '--set1_name', type=str, default='set1',
        help='Name used for set1 in CSV column headers (default: set1)'
    )
    parser.add_argument(
        '--set2_name', type=str, default='set2',
        help='Name used for set2 in CSV column headers (default: set2)'
    )
    parser.add_argument(
        '--output', type=str, default='parity_summary.csv',
        help='Output summary CSV path'
    )
    return parser.parse_args()


def load_csv(filepath):
    with open(filepath, newline='') as f:
        return list(csv.DictReader(f))


def detect_columns(headers, set1_name, set2_name):
    s1_status = f'status_{set1_name}'
    s2_status = f'status_{set2_name}'
    s1_time = f'running_time_{set1_name}'
    s2_time = f'running_time_{set2_name}'
    if s1_status not in headers:
        s1_status = 'status_set1'
        s2_status = 'status_set2'
        s1_time = 'running_time_set1'
        s2_time = 'running_time_set2'
    return s1_status, s2_status, s1_time, s2_time


def workflow_stats_keys(s1_name, s2_name):
    s1 = s1_name.upper()
    s2 = s2_name.upper()
    return [
        f'SKIPPED (on {s1_name}, but not on {s2_name})',
        f'SKIPPED (on {s1_name})',
        f'SKIPPED (on {s2_name})',
        f'MISSED (MISSED on {s1_name}, NOT SKIPPED on {s2_name})',
        f'{s1}ONLY (PASSED on {s1}, NOT PASSED on {s2})',
        s2,
        s1,
        'SKIPPED + MISSED',
        f'{s2} - (SKIPPED + MISSED)',
        f'DISAGREE [(SKIPPED+MISSED)/{s2}] %',
    ]


def compute_workflow_stats(rows, s1_col, s2_col, s1_name, s2_name):
    s1 = s1_name.upper()
    s2 = s2_name.upper()

    s1_skip_not_s2 = sum(
        1 for r in rows
        if r[s1_col] == 'SKIPPED' and r[s2_col] != 'SKIPPED'
    )
    s1_skip = sum(1 for r in rows if r[s1_col] == 'SKIPPED')
    s2_skip = sum(1 for r in rows if r[s2_col] == 'SKIPPED')
    s1_miss_not_s2_skip = sum(
        1 for r in rows
        if r[s1_col] == 'MISSED' and r[s2_col] != 'SKIPPED'
    )
    only_s1 = sum(
        1 for r in rows
        if r[s1_col] == 'PASSED' and r[s2_col] != 'PASSED'
    )
    total_s2 = sum(1 for r in rows if r[s2_col].strip())
    total_s1 = sum(1 for r in rows if r[s1_col].strip())

    skip_miss = s1_skip_not_s2 + s1_miss_not_s2_skip
    s2_minus = total_s2 - skip_miss
    pct = (skip_miss / total_s2 * 100) if total_s2 else 0

    vals = {}
    keys = workflow_stats_keys(s1_name, s2_name)
    vals[keys[0]] = s1_skip_not_s2
    vals[keys[1]] = s1_skip
    vals[keys[2]] = s2_skip
    vals[keys[3]] = s1_miss_not_s2_skip
    vals[keys[4]] = only_s1
    vals[keys[5]] = total_s2
    vals[keys[6]] = total_s1
    vals[keys[7]] = skip_miss
    vals[keys[8]] = s2_minus
    vals[keys[9]] = f'{pct:.2f}%'
    return vals


def overall_stats_keys(s1_name, s2_name):
    s1 = s1_name.upper()
    s2 = s2_name.upper()
    keys = [
        'Overall DISAGREE%',
        'Overall AGREE%',
    ]
    for status in ['PASSED', 'SKIPPED', 'FAILED', 'XFAILED']:
        keys.append(f'{status}({s1_name})')
        keys.append(f'{status}({s2_name})')
    keys += [
        f'TOTAL {s2}',
        'Number of tests changed from last week',
        f'TOTAL {s1}',
        f'TOTAL {s1} RUNNING TIME',
        f'TOTAL {s2} RUNNING TIME',
    ]
    return keys


def compute_overall_stats(rows, s1_col, s2_col, s1_time_col, s2_time_col, s1_name, s2_name):
    s1 = s1_name.upper()
    s2 = s2_name.upper()

    total_disagree = 0
    total_s2 = 0
    for wf in WORKFLOWS:
        wf_rows = [r for r in rows if r['work_flow_name'] == wf]
        s1_skip_not_s2 = sum(
            1 for r in wf_rows
            if r[s1_col] == 'SKIPPED' and r[s2_col] != 'SKIPPED'
        )
        s1_miss_not_s2_skip = sum(
            1 for r in wf_rows
            if r[s1_col] == 'MISSED' and r[s2_col] != 'SKIPPED'
        )
        total_disagree += s1_skip_not_s2 + s1_miss_not_s2_skip
        total_s2 += sum(1 for r in wf_rows if r[s2_col].strip())

    disagree_pct = (total_disagree / total_s2 * 100) if total_s2 else 0
    agree_pct = 100 - disagree_pct

    vals = {}
    keys = overall_stats_keys(s1_name, s2_name)
    vals[keys[0]] = f'{disagree_pct:.2f}%'
    vals[keys[1]] = f'{agree_pct:.2f}%'

    idx = 2
    for status in ['PASSED', 'SKIPPED', 'FAILED', 'XFAILED']:
        vals[keys[idx]] = sum(1 for r in rows if r[s1_col] == status)
        vals[keys[idx + 1]] = sum(1 for r in rows if r[s2_col] == status)
        idx += 2

    vals[keys[idx]] = sum(1 for r in rows if r[s2_col].strip())
    idx += 1
    vals[keys[idx]] = sum(
        1 for r in rows if r.get('existed_last_week', '') == 'no'
    )
    idx += 1
    vals[keys[idx]] = sum(1 for r in rows if r[s1_col].strip())
    idx += 1

    def safe_float(v):
        try:
            return float(v)
        except (ValueError, TypeError):
            return 0.0

    vals[keys[idx]] = f'{sum(safe_float(r[s1_time_col]) for r in rows):.2f}'
    idx += 1
    vals[keys[idx]] = f'{sum(safe_float(r[s2_time_col]) for r in rows):.2f}'
    return vals



def main():
    args = parse_args()

    if len(args.csv) != len(args.arch):
        print('Error: --csv and --arch must have the same number of values')
        sys.exit(1)

    archs = args.arch
    arch_data = {}
    for csv_path, arch in zip(args.csv, archs):
        rows = load_csv(csv_path)
        headers = set(rows[0].keys()) if rows else set()
        cols = detect_columns(headers, args.set1_name, args.set2_name)
        arch_data[arch] = {'rows': rows, 'cols': cols}

    out = []

    # ── Header ──
    out.append([''] + archs)
    if args.sha:
        out.append(['Commit SHA'] + [args.sha] * len(archs))
    if args.pr_id:
        out.append(['PR ID'] + [args.pr_id] * len(archs))
    out.append(
        ['GPU (MI)'] + [a.replace('mi', '').replace('MI', '') for a in archs]
    )
    out.append([])

    # ── Per-workflow stats ──
    wf_keys = workflow_stats_keys(args.set1_name, args.set2_name)
    for wf in WORKFLOWS:
        out.append([WORKFLOW_DISPLAY[wf]])
        arch_stats = {}
        for arch in archs:
            d = arch_data[arch]
            s1_col, s2_col, _, _ = d['cols']
            wf_rows = [r for r in d['rows'] if r['work_flow_name'] == wf]
            arch_stats[arch] = compute_workflow_stats(
                wf_rows, s1_col, s2_col, args.set1_name, args.set2_name
            )
        for key in wf_keys:
            out.append([key] + [arch_stats[a][key] for a in archs])
        out.append([])

    # ── Overall stats ──
    ov_keys = overall_stats_keys(args.set1_name, args.set2_name)
    arch_overall = {}
    for arch in archs:
        d = arch_data[arch]
        s1_col, s2_col, s1_time, s2_time = d['cols']
        arch_overall[arch] = compute_overall_stats(
            d['rows'], s1_col, s2_col, s1_time, s2_time,
            args.set1_name, args.set2_name,
        )
    for key in ov_keys:
        out.append([key] + [arch_overall[a][key] for a in archs])
    if args.sha:
        out.append(['Commit ID Used'] + [args.sha] * len(archs))
    out.append([])

    with open(args.output, 'w', newline='') as f:
        csv.writer(f).writerows(out)
    print(f'Summary written to {args.output}')


if __name__ == '__main__':
    main()
