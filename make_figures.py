"""Make the three website figures in one consistent style.

Usage (from the project root, after scripts/run_sql.py and scripts/run_lca.py):
    python scripts/make_figures.py

Creates in outputs/figures/ (each as .png for the web and .svg for print):
    fig1_late_by_reason   late-payment rate by main reason for using BNPL
    fig2_type_profiles    heatmap of answers by user type
    fig3_late_by_type     late-payment rate by user type

Type names are assigned from the model's results, not by class number:
the type most likely to have $400 in cash is "Cushioned users", the
remaining type most likely to say BNPL was the only way to afford it is
"Constrained borrowers", and the last is "Stretched spreaders". So the
labels stay correct even if a rerun numbers the classes differently.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

SOURCE_NOTE = ("Source: Federal Reserve, Survey of Household Economics and Decisionmaking 2025; "
               "author's calculations.\nSurvey-weighted estimates among 2,004 adults who used BNPL "
               "in the prior 12 months.")
CI_NOTE = " Whiskers show 95% confidence intervals."

TYPE_ORDER = ["Constrained borrowers", "Stretched spreaders", "Cushioned users"]
TYPE_COLORS = {"Constrained borrowers": "#B5452F",
               "Stretched spreaders": "#D9A441",
               "Cushioned users": "#3D7A8C"}
NEUTRAL = "#5B6770"

# Readable names and a logical order for the heatmap rows.
VARIABLES = {
    "cash_400": ("Can cover $400 in cash", ["Yes", "No"]),
    "emerg_3mo": ("Has 3 months of savings", ["Yes", "No"]),
    "cc_revolve": ("Credit card balance", ["Never revolves", "Sometimes revolves",
                                           "Usually revolves", "No credit card"]),
    "income_vol": ("Monthly income", ["Stable", "Occasionally varies", "Often varies"]),
    "wellbeing": ("Self-rated finances", ["Living comfortably", "Doing okay",
                                          "Just getting by", "Finding it difficult to get by"]),
    "bnpl_reason": ("Main reason for BNPL", ["Afford / no other option", "Spread payments",
                                             "Convenience", "Avoid interest", "Avoid credit card"]),
    "risk_tol": ("Willingness to take risk", ["Low (0-3)", "Medium (4-6)", "High (7-10)"]),
}


def setup_style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#888888",
        "xtick.color": "#444444",
        "ytick.color": "#444444",
        "svg.fonttype": "none",  # keep text as text in SVG
    })


def save(fig, outdir: Path, name: str, ci: bool = True) -> None:
    note = SOURCE_NOTE + (CI_NOTE if ci else "")
    fig.text(0.01, 0.005, note, fontsize=7.5, color="#666666", ha="left", va="bottom")
    for ext in ("png", "svg"):
        fig.savefig(outdir / f"{name}.{ext}", dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {name}.png / .svg")


def weighted_share(y: np.ndarray, w: np.ndarray) -> tuple[float, float, float]:
    """Weighted mean of a 0/1 variable with a 95% CI using Kish effective n."""
    p = float(w @ y / w.sum())
    n_eff = w.sum() ** 2 / (w ** 2).sum()
    se = np.sqrt(p * (1 - p) / n_eff)
    return p, p - 1.96 * se, p + 1.96 * se


def name_types(profiles: pd.DataFrame) -> dict[int, str]:
    """Map class numbers to type names using what each class looks like."""
    classes = sorted(profiles["cls"].unique())
    if len(classes) != 3:
        return {k: f"Type {k}" for k in classes}

    def prob(var, cat):
        sub = profiles[(profiles.variable == var) & (profiles.category == cat)]
        return sub.set_index("cls")["prob"]

    cushioned = prob("cash_400", "Yes").idxmax()
    afford = prob("bnpl_reason", "Afford / no other option").drop(cushioned)
    constrained = afford.idxmax()
    stretched = [k for k in classes if k not in (cushioned, constrained)][0]
    return {cushioned: "Cushioned users", constrained: "Constrained borrowers",
            stretched: "Stretched spreaders"}


# ----------------------------------------------------------------- Figure 1
def fig1_late_by_reason(df: pd.DataFrame, outdir: Path) -> None:
    d = df.dropna(subset=["bnpl_reason", "paid_late"])
    rows = []
    for reason, g in d.groupby("bnpl_reason"):
        p, lo, hi = weighted_share(g["paid_late"].to_numpy(float), g["weight"].to_numpy(float))
        rows.append(dict(reason=reason, p=p, lo=lo, hi=hi, n=len(g)))
    t = pd.DataFrame(rows).sort_values("p")
    overall, _, _ = weighted_share(d["paid_late"].to_numpy(float), d["weight"].to_numpy(float))

    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    y = np.arange(len(t))
    colors = [TYPE_COLORS["Constrained borrowers"] if r.startswith("Afford") else NEUTRAL
              for r in t["reason"]]
    ax.barh(y, t["p"] * 100, color=colors, height=0.6)
    ax.errorbar(t["p"] * 100, y, xerr=[(t["p"] - t["lo"]) * 100, (t["hi"] - t["p"]) * 100],
                fmt="none", ecolor="#222222", elinewidth=1, capsize=3)
    ax.set_yticks(y, [f"{r}  (n={n:,})" for r, n in zip(t["reason"], t["n"])])
    for yi, (p, hi) in enumerate(zip(t["p"], t["hi"])):
        ax.text(hi * 100 + 1, yi, f"{p:.0%}", va="center", fontsize=9)
    ax.axvline(overall * 100, color="#999999", ls="--", lw=1)
    ax.text(overall * 100 + 0.5, -0.65, f"All users: {overall:.0%}", fontsize=8,
            color="#666666", va="center")
    ax.set_ylim(-0.9, len(t) - 0.5)
    ax.set_xlabel("Share of BNPL users who paid late (%)")
    ax.set_xlim(0, max(t["hi"]) * 100 + 10)
    ax.set_title("Late payment by main reason for using BNPL")
    fig.subplots_adjust(bottom=0.25)
    save(fig, outdir, "fig1_late_by_reason")
    print(t.assign(p=lambda x: x.p.round(3)).to_string(index=False))


# ----------------------------------------------------------------- Figure 2
def fig2_type_profiles(profiles: pd.DataFrame, names: dict[int, str], outdir: Path) -> None:
    order = [k for name in TYPE_ORDER for k, v in names.items() if v == name] or sorted(names)
    shares = profiles.groupby("cls")["class_share"].first()

    labels, values, is_header = [], [], []
    for var, (label, cats) in VARIABLES.items():
        sub = profiles[profiles.variable == var]
        labels.append(label); values.append([np.nan] * len(order)); is_header.append(True)
        for c in [c for c in cats if c in set(sub.category)]:
            labels.append(c)
            values.append([sub[(sub.cls == k) & (sub.category == c)]["prob"].iloc[0] for k in order])
            is_header.append(False)
    vals = np.ma.masked_invalid(np.array(values, dtype=float))

    cmap = plt.get_cmap("Blues").copy()
    cmap.set_bad("white")
    fig, ax = plt.subplots(figsize=(7.5, 0.27 * len(vals) + 1.6))
    ax.imshow(vals, aspect="auto", cmap=cmap, vmin=0, vmax=1)
    ax.set_xticks(range(len(order)),
                  [f"{names[k]}\n{shares[k]:.0%} of users" for k in order], fontsize=9)
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(vals)), labels, fontsize=8.5)
    for tick, header in zip(ax.get_yticklabels(), is_header):
        if header:
            tick.set_fontweight("bold"); tick.set_color("#222222")
    for (i, j), v in np.ndenumerate(np.array(values, dtype=float)):
        if np.isfinite(v):
            ax.text(j, i, f"{v:.0%}", ha="center", va="center", fontsize=8,
                    color="white" if v > 0.55 else "#222222")
    ax.set_xticks(np.arange(-0.5, len(order)), minor=True)
    ax.grid(which="minor", axis="x", color="white", lw=2)
    ax.tick_params(length=0, which="both")
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title("Types of BNPL Users")
    save(fig, outdir, "fig2_type_profiles", ci=False)


# ----------------------------------------------------------------- Figure 3
def fig3_late_by_type(outcomes: pd.DataFrame, names: dict[int, str],
                      overall: float, outdir: Path) -> None:
    t = outcomes.assign(name=outcomes["cls"].map(names))
    t["rank"] = t["name"].map({n: i for i, n in enumerate(TYPE_ORDER)})
    t = t.sort_values("rank")

    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    x = np.arange(len(t))
    colors = [TYPE_COLORS.get(n, NEUTRAL) for n in t["name"]]
    ax.bar(x, t["mean"] * 100, color=colors, width=0.55)
    ax.errorbar(x, t["mean"] * 100,
                yerr=[(t["mean"] - t["ci_low"]) * 100, (t["ci_high"] - t["mean"]) * 100],
                fmt="none", ecolor="#222222", elinewidth=1, capsize=4)
    for xi, (m, hi) in enumerate(zip(t["mean"], t["ci_high"])):
        ax.text(xi, hi * 100 + 1.2, f"{m:.0%}", ha="center", fontsize=10, fontweight="bold",
                bbox=dict(facecolor="white", edgecolor="none", pad=1.5), zorder=4)
    ax.set_xticks(x, [f"{n}\n{s:.0%} of users" for n, s in zip(t["name"], t["class_share"])])
    ax.axhline(overall * 100, color="#999999", ls="--", lw=1, zorder=0)
    ax.text(len(t) - 0.5, overall * 100 + 0.8, f"All users: {overall:.0%}",
            fontsize=8, color="#666666", ha="right")
    ax.set_ylabel("Share who paid late (%)")
    ax.set_ylim(0, max(t["ci_high"]) * 100 + 8)
    ax.set_title("Late payment by type of BNPL user")
    fig.subplots_adjust(bottom=0.27)
    save(fig, outdir, "fig3_late_by_type")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", default=str(ROOT / "data/clean/bnpl_lca_input.csv"))
    p.add_argument("--results", default=str(ROOT / "outputs"),
                   help="Folder with lca_profiles.csv and lca_class_outcomes.csv")
    p.add_argument("--outdir", default=str(ROOT / "outputs/figures"))
    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    setup_style()

    df = pd.read_csv(args.input)
    profiles = pd.read_csv(Path(args.results) / "lca_profiles.csv")
    outcomes = pd.read_csv(Path(args.results) / "lca_class_outcomes.csv")
    names = name_types(profiles)
    print("Type names assigned:", {int(k): v for k, v in names.items()})

    d = df.dropna(subset=["paid_late"])
    overall, _, _ = weighted_share(d["paid_late"].to_numpy(float), d["weight"].to_numpy(float))

    fig1_late_by_reason(df, outdir)
    fig2_type_profiles(profiles, names, outdir)
    fig3_late_by_type(outcomes, names, overall, outdir)
    print(f"\nAll figures saved to {outdir}/")


if __name__ == "__main__":
    main()
