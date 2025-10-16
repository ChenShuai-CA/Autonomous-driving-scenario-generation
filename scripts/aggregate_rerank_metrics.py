#!/usr/bin/env python3
import os
import csv
import argparse
from collections import defaultdict

def parse_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default

def main():
    parser = argparse.ArgumentParser(description="Aggregate rerank metrics per GIF from master CSV.")
    parser.add_argument("--master", type=str, default=os.path.join("Output","rerank_master.csv"), help="Path to rerank_master.csv")
    parser.add_argument("--out", type=str, default=os.path.join("Output","rerank_summary.csv"), help="Output summary CSV path")
    parser.add_argument("--topk", type=int, default=3, help="Top-K to average for summary")
    args = parser.parse_args()

    if not os.path.exists(args.master):
        print(f"[ERR] master not found: {args.master}")
        return 1

    # group rows by gif
    groups = defaultdict(list)
    with open(args.master, "r", newline="") as f:
        reader = csv.DictReader(f)
        required = {"gif","idx","rank","score","ade","out_cnt","out_frac","jerk","selected","lambda_lane","lambda_jerk"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            print(f"[ERR] master missing columns: {sorted(missing)}")
            return 2
        for row in reader:
            # convert numeric fields
            row["rank"] = int(row["rank"]) if row["rank"] != "" else 1_000_000
            row["idx"] = int(row["idx"]) if row["idx"] != "" else -1
            row["score"] = parse_float(row["score"])
            row["ade"] = parse_float(row["ade"])
            row["out_cnt"] = int(float(row["out_cnt"])) if row["out_cnt"] != "" else 0
            row["out_frac"] = parse_float(row["out_frac"])  # already normalized
            row["jerk"] = parse_float(row["jerk"])
            row["selected"] = int(float(row["selected"])) if row["selected"] != "" else 0
            row["lambda_lane"] = parse_float(row["lambda_lane"], default=float("nan"))
            row["lambda_jerk"] = parse_float(row["lambda_jerk"], default=float("nan"))
            groups[row["gif"]].append(row)

    # write summary
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "gif","n_candidates","topk",
            "best_score","best_ade","best_out_frac","best_jerk",
            "topk_score_mean","topk_ade_mean","topk_out_frac_mean","topk_jerk_mean",
            "selected_in_topk","lambda_lane","lambda_jerk"
        ])
        for gif, rows in groups.items():
            rows_sorted = sorted(rows, key=lambda r: r["rank"])
            k = min(args.topk, len(rows_sorted)) if len(rows_sorted) > 0 else 0
            if k == 0:
                continue
            best = rows_sorted[0]
            topk_rows = rows_sorted[:k]
            def avg(key):
                vals = [parse_float(r[key]) for r in topk_rows]
                return sum(vals)/len(vals) if vals else 0.0
            selected_in_topk = sum(r["selected"] for r in topk_rows)
            writer.writerow([
                gif,
                len(rows_sorted),
                k,
                f"{best['score']:.6f}", f"{best['ade']:.6f}", f"{best['out_frac']:.6f}", f"{best['jerk']:.6f}",
                f"{avg('score'):.6f}", f"{avg('ade'):.6f}", f"{avg('out_frac'):.6f}", f"{avg('jerk'):.6f}",
                selected_in_topk,
                best.get('lambda_lane',''), best.get('lambda_jerk','')
            ])
    print(f"[AGG] summary saved: {args.out}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
