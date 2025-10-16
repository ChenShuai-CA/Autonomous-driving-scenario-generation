import os
import csv
import argparse
from typing import List

import matplotlib.pyplot as plt

def read_csv(path: str):
    rows = []
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV not found: {path}")
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows

def to_float(val):
    try:
        return float(val)
    except Exception:
        return None

def main(args):
    rows = read_csv(args.csv)
    if not rows:
        print("Empty CSV")
        return
    # collect epochs
    epochs = [int(r['epoch']) for r in rows if 'epoch' in r and r['epoch']]
    # find component weight columns
    sample_row = rows[-1]
    compw_cols = [c for c in sample_row.keys() if c.startswith('compw_') and not c.endswith('entropy')]
    compw_cols = sorted(compw_cols)
    if not compw_cols:
        print("No component weight columns found (compw_*)")
    # risk global & ema
    rgs = [to_float(r.get('risk_global_scale')) for r in rows]
    ema_base = [to_float(r.get('risk_autoscale_ema_base')) for r in rows]
    ema_raw = [to_float(r.get('risk_autoscale_ema_raw_score')) for r in rows]
    entropy_lambda = [to_float(r.get('compw_entropy_lambda')) for r in rows]

    # ---- Plot component weights ----
    if compw_cols:
        plt.figure(figsize=(8,4))
        for c in compw_cols:
            vals = [to_float(r.get(c)) for r in rows]
            plt.plot(epochs, vals, label=c)
        plt.xlabel('Epoch'); plt.ylabel('Weight'); plt.title('Learned Component Weights')
        plt.legend(); plt.tight_layout()
        if args.out_prefix:
            plt.savefig(args.out_prefix + '_compw.png', dpi=160)
        if not args.no_show:
            plt.show()
        plt.close()

    # ---- Plot autoscale dynamics ----
    plt.figure(figsize=(8,4))
    plt.plot(epochs, rgs, label='risk_global_scale')
    if any(e is not None for e in ema_base):
        plt.plot(epochs, ema_base, label='ema_base')
    if any(e is not None for e in ema_raw):
        plt.plot(epochs, ema_raw, label='ema_raw_risk')
    plt.xlabel('Epoch'); plt.ylabel('Value'); plt.title('Autoscale Dynamics')
    plt.legend(); plt.tight_layout()
    if args.out_prefix:
        plt.savefig(args.out_prefix + '_autoscale.png', dpi=160)
    if not args.no_show:
        plt.show()
    plt.close()

    # ---- Plot entropy lambda (if constant still draws) ----
    if any(e is not None for e in entropy_lambda):
        plt.figure(figsize=(6,3))
        plt.plot(epochs, entropy_lambda, label='compw_entropy_lambda')
        plt.xlabel('Epoch'); plt.ylabel('λ'); plt.title('Entropy Regularization Coefficient')
        plt.legend(); plt.tight_layout()
        if args.out_prefix:
            plt.savefig(args.out_prefix + '_entropy_lambda.png', dpi=160)
        if not args.no_show:
            plt.show()
        plt.close()
    print("Done. Saved plots with prefix" if args.out_prefix else "Done.")

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', default='log_rebound/autoscale_log.csv')
    ap.add_argument('--out-prefix', default='log_rebound/compw')
    ap.add_argument('--no-show', action='store_true')
    args = ap.parse_args()
    main(args)
