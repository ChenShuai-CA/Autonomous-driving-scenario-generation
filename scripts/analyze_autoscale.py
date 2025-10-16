#!/usr/bin/env python3
"""Analyze autoscale_log.csv convergence behavior.

Usage:
  python scripts/analyze_autoscale.py --log log_rebound_norm_full/autoscale_log.csv [--tol 0.1 0.05 0.02]

Outputs:
  * Basic stats: first/last scale, final target fraction, epochs
  * Convergence epochs for each tolerance (earliest epoch where |scale_final - scale_epoch|/scale_final <= tol)
  * Estimated asymptotic scale (median of last 10% epochs)
  * Effective scale growth ratio per epoch (geometric) initial->asymptotic
"""
import argparse, csv, math, statistics, sys
from pathlib import Path

def load_rows(path):
    rows = []
    with open(path, 'r') as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                row['epoch'] = int(row['epoch'])
                # cast numeric fields
                for k,v in list(row.items()):
                    if k == 'epoch':
                        continue
                    try:
                        row[k] = float(v)
                    except Exception:
                        pass
                rows.append(row)
            except Exception:
                continue
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--log', required=True, help='Path to autoscale_log.csv')
    ap.add_argument('--tol', nargs='*', type=float, default=[0.10, 0.05, 0.02], help='Relative tolerance levels to report convergence epochs')
    args = ap.parse_args()
    p = Path(args.log)
    if not p.exists():
        print(f"[ERR] Log file not found: {p}")
        sys.exit(1)
    rows = load_rows(p)
    if not rows:
        print('[ERR] No rows parsed.')
        sys.exit(1)

    epochs = [r['epoch'] for r in rows]
    scales = [r['risk_global_scale'] for r in rows]
    ema_base = [r.get('risk_autoscale_ema_base') for r in rows]
    ema_raw = [r.get('risk_autoscale_ema_raw_score') for r in rows]
    target_frac = rows[-1].get('risk_autoscale_target_frac')

    final_scale = scales[-1]
    # Last 10% epochs median as asymptotic estimate
    tail_k = max(1, len(scales)//10)
    asymptotic = statistics.median(scales[-tail_k:])

    print('=== Autoscale Convergence Summary ===')
    print(f' Log file            : {p}')
    print(f' Epochs              : {epochs[0]}..{epochs[-1]} (count={len(epochs)})')
    print(f' Initial scale       : {scales[0]:.4g}')
    print(f' Final scale         : {final_scale:.4g}')
    print(f' Asymptotic (median) : {asymptotic:.4g} (tail window={tail_k})')
    if scales[0] > 0 and asymptotic > 0 and len(scales) > 1:
        growth_factor = asymptotic / scales[0]
        per_epoch_geo = growth_factor ** (1/(len(scales)-1))
        print(f' Geometric growth per epoch ~ {per_epoch_geo:.4f}')
    if target_frac is not None:
        print(f' Target fraction     : {target_frac}')

    for tol in args.tol:
        conv_epoch = None
        for e, s in zip(epochs, scales):
            if final_scale == 0:
                continue
            rel_err = abs(final_scale - s) / final_scale
            if rel_err <= tol:
                conv_epoch = e
                break
        if conv_epoch is None:
            status = 'NOT REACHED'
        else:
            status = f'epoch {conv_epoch} (Δ={conv_epoch-epochs[0]})'
        print(f' Convergence tol {tol*100:.1f}% : {status}')

    # Optional: print last few rows for inspection
    print('\nLast 5 rows:')
    for r in rows[-5:]:
        print(f"  epoch={r['epoch']}, scale={r['risk_global_scale']:.4g}, ema_base={r.get('risk_autoscale_ema_base'):.4g}, ema_raw={r.get('risk_autoscale_ema_raw_score'):.4g}")

if __name__ == '__main__':
    main()
