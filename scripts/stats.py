"""Friedman test + paired Wilcoxon (hybrid vs each baseline) on the per-slice
metrics written by evaluate.py. Holm-adjusted p-values are reported alongside
the raw ones.

Example:
    python scripts/stats.py results/metrics.csv --hybrid Hybrid_Deployable
"""

import argparse
import csv

import numpy as np
from scipy import stats

BASELINES = ["Air", "Bone", "SwinIR"]


def holm(pvals):
    """Holm-Bonferroni step-down adjusted p-values."""
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * pvals[idx])
        adj[idx] = min(1.0, running)
    return adj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_path")
    ap.add_argument("--hybrid", default="Hybrid_Deployable",
                    help="column suffix of the hybrid to test (Hybrid_Oracle | Hybrid_Deployable)")
    ap.add_argument("--metrics", nargs="+", default=["PSNR", "SSIM", "FSIM"])
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    with open(args.csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    col = lambda name: np.array([float(r[name]) for r in rows])  # noqa: E731

    methods = BASELINES + [args.hybrid]
    print(f"n = {len(rows)} slices | hybrid = {args.hybrid}\n")

    table, raw_all = [], []
    for metric in args.metrics:
        data = [col(f"{metric}_{m}") for m in methods]
        fr_p = stats.friedmanchisquare(*data).pvalue
        hy = data[-1]
        for base, d in zip(BASELINES, data[:-1]):
            p_raw = stats.wilcoxon(hy, d).pvalue
            table.append([metric, base, fr_p, p_raw, float(np.mean(hy - d)), float(np.median(hy - d))])
            raw_all.append(p_raw)

    adj = holm(np.array(raw_all))
    for row, a in zip(table, adj):
        row.append(float(a))

    header = ["metric", "vs", "friedman_p", "wilcoxon_p", "mean_diff", "median_diff", "holm_p"]
    print(f"{header[0]:<7}{header[1]:<8}{header[2]:>12}{header[3]:>13}{header[4]:>11}{header[5]:>13}{header[6]:>11}")
    for r in table:
        print(f"{r[0]:<7}{r[1]:<8}{r[2]:>12.2e}{r[3]:>13.2e}{r[4]:>11.4f}{r[5]:>13.4f}{r[6]:>11.2e}")
    print("\nmean_diff / median_diff = hybrid minus baseline (positive favours the hybrid).")

    if args.out:
        with open(args.out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(table)
        print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
