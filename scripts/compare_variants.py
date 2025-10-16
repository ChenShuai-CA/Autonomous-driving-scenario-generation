#!/usr/bin/env python3
"""Quick A/B comparison for model variants (baseline GRU vs Mamba+MHA).

Usage (example):
  python scripts/compare_variants.py \
    --configs config/Interaction_quick.py config/Interaction_quick_mamba.py \
    --label GRU --label MambaMHA

This launches short runs (epochs/batches defined in each config), logs basic metrics
(ADE/FDE if exposed, risk stats) and prints a comparison table at the end.

Assumptions:
- main.py exposes a train() entry triggered via importing or by running main as script.
- A function get_last_metrics() or similar is NOT present; we therefore parse autoscale_log.csv
  and optionally a produced metrics npz/pt if available. If not, we only collect risk scaling stats.

If deeper integration is desired, consider refactoring main.py to return a metrics dict.
"""
from __future__ import annotations
import argparse
import importlib.util
import os
import json
import sys
import time
import csv
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'Code'))

def load_config_module(path: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location("_ab_config", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load config: {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore
    return mod

def run_variant(config_path: str, label: str):
    print(f"\n=== Running variant: {label} ({config_path}) ===")
    # Set an env var so main can optionally shorten logging if implemented later
    os.environ['AB_LABEL'] = label
    start = time.time()
    # Dynamically set CONFIG_PATH environment so main.py (if it reads) can pick it up; else user must manually adapt main
    os.environ['CONFIG_PATH_OVERRIDE'] = config_path
    # We assume main.py uses a pattern like: from config import Interaction_quick as C
    # For flexibility, we just exec main in a new global namespace after adjusting sys.modules.
    main_path = ROOT / 'Code' / 'main.py'
    with open(main_path, 'r', encoding='utf-8') as f:
        code = f.read()
    # Provide globals with a hook variable to signal alternative config; user can modify main.py to check it.
    g = {'__name__': '__ab_run__'}
    # Optionally monkeypatch config import by preloading selected config under a canonical name if main uses e.g., from config import Interaction_quick as cfg
    # We attempt heuristic: if filename contains 'quick_mamba', alias to Interaction_quick_mamba; else interaction_quick.
    cfg_mod = load_config_module(config_path)
    base_name = Path(config_path).stem
    sys.modules[f'config.{base_name}'] = cfg_mod
    # Provide a canonical alias 'config.current_ab_cfg' for potential future use
    sys.modules['config.current_ab_cfg'] = cfg_mod
    try:
        exec(compile(code, str(main_path), 'exec'), g)
    except SystemExit:
        # Allow main to call sys.exit
        pass
    dur = time.time() - start
    # After run, inspect autoscale_log.csv if present
    autoscale_csv = ROOT / 'log_rebound' / 'autoscale_log.csv'
    risk_rows = []
    if autoscale_csv.exists():
        with autoscale_csv.open('r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                risk_rows.append(row)
    metrics = {}
    if risk_rows:
        last = risk_rows[-1]
        for k in ['risk_score_raw','risk_score','risk_scaled','log_sigma_penalty']:
            if k in last:
                try:
                    metrics[k] = float(last[k])
                except ValueError:
                    pass
        # Basic stability: std of scaled risk if available
        try:
            rs_vals = [float(r['risk_scaled']) for r in risk_rows if 'risk_scaled' in r]
            if len(rs_vals) > 2:
                import statistics
                metrics['risk_scaled_std'] = statistics.pstdev(rs_vals)
        except Exception:
            pass
    metrics['duration_sec'] = dur
    # Load A/B JSON metrics if produced by main.py
    ckpt_dir = ROOT / 'log_rebound'
    json_path = ckpt_dir / f'ab_metrics_{label}.json'
    if json_path.exists():
        try:
            with open(json_path, 'r') as jf:
                abm = json.load(jf)
            metrics.update({k: v for k, v in abm.items() if k not in metrics})
        except Exception as e:
            print(f"[WARN] Failed loading metrics JSON for {label}: {e}")
    return metrics

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--configs', nargs='+', required=True, help='List of config file paths.')
    ap.add_argument('--label', nargs='+', required=True, help='Human-readable labels for variants (same length).')
    args = ap.parse_args()

    if len(args.configs) != len(args.label):
        raise SystemExit('configs and label lengths differ')

    results = []
    for cfg, lbl in zip(args.configs, args.label):
        metrics = run_variant(cfg, lbl)
        results.append((lbl, metrics))

    # Print summary table
    keys = sorted({k for _, m in results for k in m.keys()})
    print('\n=== A/B Summary ===')
    header = ['label'] + keys
    colw = [max(len(h), 12) for h in header]
    def fmt_row(vals):
        return ' | '.join(f"{v:>{w}}" for v, w in zip(vals, colw))
    print(fmt_row(header))
    print('-+-'.join('-'*w for w in colw))
    for lbl, m in results:
        row = [lbl]
        for k in keys:
            v = m.get(k, '')
            if isinstance(v, float):
                row.append(f"{v:.4g}")
            else:
                row.append(str(v))
        print(fmt_row(row))

if __name__ == '__main__':
    main()
