"""Fit the latent class model to 2025 BNPL users and save results.

Usage (from the project root, after scripts/run_sql.py):
    python scripts/run_lca.py                    # compare K = 1..6, pick by BIC
    python scripts/run_lca.py --k 4              # force 4 classes
    python scripts/run_lca.py --drop-speeders --drop-imputed   # robustness run

Outputs (in outputs/ or --outdir):
    lca_fit_stats.csv        fit statistics for each K (BIC, AIC, entropy, ...)
    lca_profiles.csv         P(answer | class) for every indicator
    lca_class_outcomes.csv   weighted late-payment rate by class
    lca_posteriors.csv       each user's class probabilities
    fig_bic.png, fig_profiles.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from shed_bnpl.lca import compare_k, distal_outcome_by_class  # noqa: E402

INDICATORS = ["cash_400", "emerg_3mo", "cc_revolve", "income_vol",
              "wellbeing", "bnpl_reason", "risk_tol"]
OUTCOME = "paid_late"


def load(path: Path, drop_speeders: bool, drop_imputed: bool) -> pd.DataFrame:
    df = pd.read_csv(path)
    n0 = len(df)
    for flag, drop in (("speeder", drop_speeders), ("any_imputed", drop_imputed)):
        if drop:
            df = df[~df[flag].astype(str).str.lower().isin(["true", "1"])]
    print(f"Loaded {n0} BNPL users; using {len(df)} after filters.")
    return df.reset_index(drop=True)


def plot_bic(stats: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(stats["K"], stats["BIC"], marker="o", label="BIC")
    ax.plot(stats["K"], stats["AIC"], marker="s", label="AIC")
    ax.set_xlabel("Number of classes (K)")
    ax.set_ylabel("Information criterion (lower = better)")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_profiles(prof: pd.DataFrame, path: Path) -> None:
    """Heatmap: rows = indicator categories, columns = classes."""
    import matplotlib.pyplot as plt
    prof = prof.copy()
    prof["row"] = prof["variable"] + ": " + prof["category"].astype(str)
    wide = prof.pivot_table(index="row", columns="cls", values="prob", sort=False)
    shares = prof.groupby("cls")["class_share"].first()

    fig, ax = plt.subplots(figsize=(2.2 + 1.4 * wide.shape[1], 0.32 * len(wide) + 1.5))
    im = ax.imshow(wide.values, aspect="auto", cmap="Blues", vmin=0, vmax=1)
    ax.set_yticks(range(len(wide)), wide.index, fontsize=8)
    ax.set_xticks(range(wide.shape[1]),
                  [f"Class {k}\n({shares[k]:.0%})" for k in wide.columns], fontsize=9)
    for (i, j), v in np.ndenumerate(wide.values):
        ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                color="white" if v > 0.6 else "black")
    fig.colorbar(im, ax=ax, fraction=0.03, label="P(answer | class)")
    ax.set_title("BNPL user types: answer probabilities by latent class", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", default=str(ROOT / "data/clean/bnpl_lca_input.csv"))
    p.add_argument("--outdir", default=str(ROOT / "outputs"))
    p.add_argument("--k-max", type=int, default=6)
    p.add_argument("--k", type=int, help="Force this number of classes")
    p.add_argument("--n-init", type=int, default=30)
    p.add_argument("--seed", type=int, default=2025)
    p.add_argument("--drop-speeders", action="store_true")
    p.add_argument("--drop-imputed", action="store_true")
    args = p.parse_args()

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    df = load(Path(args.input), args.drop_speeders, args.drop_imputed)
    X, w = df[INDICATORS], df["weight"].to_numpy()

    stats, models = compare_k(X, w, range(1, args.k_max + 1),
                              n_init=args.n_init, random_state=args.seed)
    stats.to_csv(out / "lca_fit_stats.csv", index=False)
    print("\nModel comparison:\n" + stats.round(3).to_string(index=False))

    k = args.k or int(stats.loc[stats["BIC"].idxmin(), "K"])
    model = models[k]
    print(f"\nUsing K = {k} ({'forced' if args.k else 'lowest BIC'}).")
    if model.pi_.min() < 0.05:
        print("  Warning: smallest class is under 5% of users; consider a smaller K.")

    prof = model.profiles()
    prof.to_csv(out / "lca_profiles.csv", index=False)

    resp = model.predict_proba(X)
    post = pd.DataFrame(resp, columns=[f"p_class{c + 1}" for c in range(k)])
    post.insert(0, "shedid", df["shedid"])
    post["modal_class"] = resp.argmax(axis=1) + 1
    post.to_csv(out / "lca_posteriors.csv", index=False)

    outcomes = distal_outcome_by_class(resp, df[OUTCOME], w)
    outcomes.insert(1, "class_share", model.pi_)
    outcomes.to_csv(out / "lca_class_outcomes.csv", index=False)
    print("\nLate-payment rate by class (weighted, soft assignment):")
    print(outcomes.round(3).to_string(index=False))

    print("\nMost likely answer for each indicator, by class:")
    top = (prof.sort_values("prob", ascending=False)
               .groupby(["cls", "variable"]).head(1)
               .assign(answer=lambda d: d["category"].astype(str) + f" (" + d["prob"].round(2).astype(str) + ")")
               .pivot(index="variable", columns="cls", values="answer")
               .reindex(INDICATORS))
    print(top.to_string())

    try:
        plot_bic(stats, out / "fig_bic.png")
        plot_profiles(prof, out / "fig_profiles.png")
        print(f"\nFigures saved to {out}/")
    except ImportError:
        print("\nmatplotlib not installed; skipped figures.")


if __name__ == "__main__":
    main()
