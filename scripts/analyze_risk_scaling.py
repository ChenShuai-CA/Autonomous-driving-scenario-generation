import re
import sys
from pathlib import Path
from statistics import mean
from collections import defaultdict
"""
Quick parser for training stdout logs to summarize risk scaling behavior across different risk_global_scale settings.

Usage:
  1) Redirect your training output to a file, e.g.:
       python Code/main.py --train ... > run_risk1e3.log 2>&1
  2) Then run:
       python scripts/analyze_risk_scaling.py run_risk1e3.log [run_risk1e5.log ...]
  3) The script will summarize per-file (treated as one scale) statistics:
       - mean risk_score
       - mean risk_scaled
       - mean risk_contrib
       - mean comp_risk_overlap
       - overlap dominance ratio (comp_risk_overlap / risk_score)
       - inferred risk_global_scale (first occurrence)

If multiple risk_global_scale values appear in one file, each line is grouped by that scale.
You can also pipe a single log via stdin:
       python scripts/analyze_risk_scaling.py - < training.log

Limitations:
 - Relies on consistent key pattern "risk_score:" etc.
 - Skips lines without required numeric fields.
"""

# Regex patterns for extracting metrics
PATTERN = re.compile(
    r"risk_score:\s*([0-9eE+\.-]+).*?risk_scaled:\s*([0-9eE+\.-]+).*?risk_weight:\s*([0-9eE+\.-]+).*?risk_global_scale:\s*([0-9eE+\.-]+).*?risk_contrib:\s*([0-9eE+\.-]+).*?comp_risk_overlap:\s*([0-9eE+\.-]+)")

def parse_stream(stream):
    groups = defaultdict(lambda: defaultdict(list))  # scale -> metric -> list
    for line in stream:
        m = PATTERN.search(line)
        if not m:
            continue
        try:
            risk_score, risk_scaled, risk_weight, risk_scale, risk_contrib, comp_overlap = map(float, m.groups())
        except ValueError:
            continue
        g = groups[risk_scale]
        g['risk_score'].append(risk_score)
        g['risk_scaled'].append(risk_scaled)
        g['risk_weight'].append(risk_weight)
        g['risk_contrib'].append(risk_contrib)
        g['comp_risk_overlap'].append(comp_overlap)
    return groups

def summarize(groups):
    rows = []
    for scale, metrics in sorted(groups.items(), key=lambda x: x[0]):
        if not metrics['risk_score']:
            continue
        rs_mean = mean(metrics['risk_score'])
        rscaled_mean = mean(metrics['risk_scaled'])
        contrib_mean = mean(metrics['risk_contrib'])
        ov_mean = mean(metrics['comp_risk_overlap'])
        # Avoid division by zero
        ov_dom = ov_mean / rs_mean if rs_mean > 1e-9 else float('nan')
        rows.append({
            'risk_global_scale': scale,
            'mean_risk_score': rs_mean,
            'mean_risk_scaled': rscaled_mean,
            'mean_risk_contrib': contrib_mean,
            'mean_comp_risk_overlap': ov_mean,
            'overlap_dominance_ratio': ov_dom,
            'n_samples': len(metrics['risk_score']),
        })
    return rows

def format_table(rows):
    if not rows:
        return "No parsed metrics."
    headers = [
        'risk_global_scale','mean_risk_score','mean_risk_scaled','mean_risk_contrib',
        'mean_comp_risk_overlap','overlap_dom_ratio','n_samples'
    ]
    colw = {h: max(len(h), 14) for h in headers}
    for r in rows:
        for h in headers:
            colw[h] = max(colw[h], len(f"{r[h]:.6g}") if isinstance(r[h], (int,float)) else len(str(r[h])))
    def fmt(r):
        return " | ".join([
            f"{r['risk_global_scale']:.6g}".ljust(colw['risk_global_scale']),
            f"{r['mean_risk_score']:.6g}".ljust(colw['mean_risk_score']),
            f"{r['mean_risk_scaled']:.6g}".ljust(colw['mean_risk_scaled']),
            f"{r['mean_risk_contrib']:.6g}".ljust(colw['mean_risk_contrib']),
            f"{r['mean_comp_risk_overlap']:.6g}".ljust(colw['mean_comp_risk_overlap']),
            f"{r['overlap_dominance_ratio']:.6g}".ljust(colw['overlap_dom_ratio']),
            f"{r['n_samples']}".ljust(colw['n_samples'])
        ])
    header_line = " | ".join([h.ljust(colw[h]) for h in headers])
    sep_line = "-+-".join(['-'*colw[h] for h in headers])
    body = "\n".join(fmt(r) for r in rows)
    return f"{header_line}\n{sep_line}\n{body}"


def main():
    paths = sys.argv[1:]
    if not paths:
        print("Provide at least one log file, or '-' for stdin.")
        return
    all_rows = []
    for p in paths:
        if p == '-':
            groups = parse_stream(sys.stdin)
            rows = summarize(groups)
            print(f"\n=== Summary: <stdin> ===")
            print(format_table(rows))
            all_rows.extend(rows)
        else:
            file_path = Path(p)
            if not file_path.exists():
                print(f"[WARN] File not found: {file_path}")
                continue
            with file_path.open('r', encoding='utf-8', errors='ignore') as f:
                groups = parse_stream(f)
            rows = summarize(groups)
            print(f"\n=== Summary: {file_path.name} ===")
            print(format_table(rows))
            all_rows.extend(rows)

    # Optional consolidated ranking by overlap dominance
    if all_rows:
        print("\n=== Consolidated (sorted by overlap dominance descending) ===")
        all_rows_sorted = sorted(all_rows, key=lambda r: r['overlap_dominance_ratio'], reverse=True)
        print(format_table(all_rows_sorted))

if __name__ == '__main__':
    main()
