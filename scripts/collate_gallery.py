#!/usr/bin/env python3
import argparse
import csv
import os
import shutil
from datetime import datetime


def read_dangerous_csv(csv_path: str):
    rows = []
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"dangerous_trajs.csv not found: {csv_path}")
    with open(csv_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for r in reader:
            try:
                r['risk_score'] = float(r.get('risk_score', 'nan'))
            except Exception:
                r['risk_score'] = float('nan')
            rows.append(r)
    return rows


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)
    return path


def copy_with_meta(rows, out_dir: str, root_out: str = "Output", topk: int = 100):
    # 以 risk_score 降序排序，并按 gif 去重
    rows_sorted = sorted(rows, key=lambda r: (r['risk_score'] if r['risk_score'] == r['risk_score'] else -1e9), reverse=True)
    seen = set()
    picked = []
    k = max(1, int(topk))
    for r in rows_sorted:
        g = r.get('gif', '')
        if not g or g in seen:
            continue
        seen.add(g)
        picked.append(r)
        if len(picked) >= k:
            break

    meta_csv = os.path.join(out_dir, 'top100_meta.csv')
    with open(meta_csv, 'w', newline='') as f:
        fieldnames = [
            'gif', 'risk_score', 'src_case_id', 'window_start', 'ego_id', 'ego_type',
            'neighbor_ids', 'neighbor_types', 'highlight_neighbor_ids'
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in picked:
            writer.writerow({k: r.get(k, '') for k in fieldnames})

    # 复制 GIF 及相关 CSV
    for r in picked:
        gif_name = r.get('gif', '')
        if not gif_name:
            continue
        base = os.path.splitext(gif_name)[0]
        for ext in ('.gif', '.csv', '_trajs.csv', '_trajs_highlight.csv'):
            src = os.path.join(root_out, base + ext)
            if os.path.exists(src):
                dst = os.path.join(out_dir, os.path.basename(src))
                try:
                    shutil.copy2(src, dst)
                except Exception as e:
                    print(f"[WARN] copy failed {src} -> {dst}: {e}")

    print(f"Copied {len(picked)} items into {out_dir}")
    print(f"Meta written: {meta_csv}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output-dir', default='Output', help='Directory where generation outputs are stored')
    ap.add_argument('--gallery-root', default='gallery', help='Root directory for gallery outputs')
    ap.add_argument('--topk', type=int, default=100, help='Top-K items to collect (<=100; script will still cap by available)')
    args = ap.parse_args()

    root = os.getcwd()
    out_dir = os.path.join(root, args.output_dir)
    gal_root = os.path.join(root, args.gallery_root)
    ensure_dir(gal_root)
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    gal_dir = ensure_dir(os.path.join(gal_root, f'top{args.topk}_{stamp}'))

    rows = read_dangerous_csv(os.path.join(out_dir, 'dangerous_trajs.csv'))
    if not rows:
        print("[INFO] No rows in dangerous_trajs.csv yet. Nothing to copy.")
        return
    copy_with_meta(rows, gal_dir, root_out=out_dir, topk=args.topk)


if __name__ == '__main__':
    main()
