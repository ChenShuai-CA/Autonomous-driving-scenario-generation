import csv, os, argparse
import matplotlib.pyplot as plt

def load_csv(path):
    rows = []
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows

def maybe_float(v):
    try:
        return float(v)
    except Exception:
        return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', type=str, required=True, help='Path to autoscale_log.csv')
    ap.add_argument('--out', type=str, default=None, help='Output figure path (png). If not set show interactively.')
    args = ap.parse_args()

    rows = load_csv(args.csv)
    if not rows:
        print('No rows in CSV.')
        return
    epochs = [int(r.get('epoch', i+1)) for i, r in enumerate(rows)]

    # Candidate columns (fallback if some missing)
    col_map = {
        'risk_score_raw': 'risk_score_raw',
        'risk_score': 'risk_score',
        'risk_scaled': 'risk_scaled',
        'log_sigma_penalty': 'log_sigma_penalty'
    }
    series = {}
    for label, col in col_map.items():
        vals = [maybe_float(r.get(col, '')) for r in rows]
        if any(v is not None for v in vals):
            series[label] = [v if v is not None else float('nan') for v in vals]

    if not series:
        print('No matching columns found in CSV (need at least one of: risk_score_raw, risk_score, risk_scaled, log_sigma_penalty).')
        return

    plt.figure(figsize=(10,5))
    for k, v in series.items():
        plt.plot(epochs, v, label=k)
    plt.xlabel('epoch')
    plt.ylabel('value')
    plt.title('Risk & log_sigma_penalty evolution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
        plt.savefig(args.out, dpi=150, bbox_inches='tight')
        print('Saved figure to', args.out)
    else:
        plt.show()

if __name__ == '__main__':
    main()
