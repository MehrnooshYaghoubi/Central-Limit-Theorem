#!/usr/bin/env python3
"""
Central Limit Theorem Demonstration

Visualizes how sample means converge to a normal distribution regardless
of the underlying population distribution (exponential, uniform, Poisson,
gamma, or log-normal).

Usage:
    python plot_sample_means.py                        # defaults
    python plot_sample_means.py -d uniform -n 50 -N 2000
    python plot_sample_means.py --animate --distribution poisson
    python plot_sample_means.py --list-distributions
"""

import argparse
import sys
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation
from scipy import stats


# ── Distribution registry ────────────────────────────────────────────────────

@dataclass
class Distribution:
    name: str
    label: str
    generator: Callable[[np.random.Generator, int], np.ndarray]
    theoretical_mean: float
    theoretical_std: float  # population σ, NOT std of sample means

    def sample_means(
        self, rng: np.random.Generator, n_samples: int, sample_size: int
    ) -> np.ndarray:
        data = self.generator(rng, n_samples * sample_size)
        return data.reshape(n_samples, sample_size).mean(axis=1)

    def theoretical_sem(self, sample_size: int) -> float:
        """Standard error of the mean: σ / √n"""
        return self.theoretical_std / np.sqrt(sample_size)


def _build_distributions() -> dict:
    return {
        "exponential": Distribution(
            name="exponential",
            label="Exponential (scale=2)",
            generator=lambda rng, size: rng.exponential(scale=2.0, size=size),
            theoretical_mean=2.0,
            theoretical_std=2.0,
        ),
        "uniform": Distribution(
            name="uniform",
            label="Uniform (0, 10)",
            generator=lambda rng, size: rng.uniform(low=0.0, high=10.0, size=size),
            theoretical_mean=5.0,
            theoretical_std=10.0 / (2.0 * np.sqrt(3.0)),
        ),
        "poisson": Distribution(
            name="poisson",
            label="Poisson (λ=5)",
            generator=lambda rng, size: rng.poisson(lam=5, size=size).astype(float),
            theoretical_mean=5.0,
            theoretical_std=np.sqrt(5.0),
        ),
        "gamma": Distribution(
            name="gamma",
            label="Gamma (k=2, θ=2)",
            generator=lambda rng, size: rng.gamma(shape=2.0, scale=2.0, size=size),
            theoretical_mean=4.0,
            theoretical_std=2.0 * np.sqrt(2.0),
        ),
        "lognormal": Distribution(
            name="lognormal",
            label="Log-Normal (μ=0, σ=1)",
            generator=lambda rng, size: rng.lognormal(mean=0.0, sigma=1.0, size=size),
            theoretical_mean=float(np.exp(0.5)),
            theoretical_std=float(np.sqrt((np.exp(1.0) - 1.0) * np.exp(1.0))),
        ),
    }


DISTRIBUTIONS = _build_distributions()


# ── Statistical analysis ─────────────────────────────────────────────────────

@dataclass
class CLTReport:
    empirical_mean: float
    empirical_std: float
    theoretical_mean: float
    theoretical_sem: float
    ks_stat: float
    ks_pvalue: float
    shapiro_stat: float
    shapiro_pvalue: float

    @property
    def normality_passed(self) -> bool:
        return self.ks_pvalue > 0.05 and self.shapiro_pvalue > 0.05


def analyse(means: np.ndarray, dist: Distribution, sample_size: int) -> CLTReport:
    empirical_mean = float(np.mean(means))
    empirical_std = float(np.std(means, ddof=1))

    ks_stat, ks_p = stats.kstest(
        means, "norm", args=(empirical_mean, empirical_std)
    )
    # Shapiro-Wilk is limited to 5 000 observations
    sw_sample = means[:5000] if len(means) > 5000 else means
    sw_stat, sw_p = stats.shapiro(sw_sample)

    return CLTReport(
        empirical_mean=empirical_mean,
        empirical_std=empirical_std,
        theoretical_mean=dist.theoretical_mean,
        theoretical_sem=dist.theoretical_sem(sample_size),
        ks_stat=float(ks_stat),
        ks_pvalue=float(ks_p),
        shapiro_stat=float(sw_stat),
        shapiro_pvalue=float(sw_p),
    )


# ── Theme helpers ─────────────────────────────────────────────────────────────

_BG_DARK  = "#1e1e2e"
_BG_PANEL = "#2d2d44"
_GRID     = "#4a4a6a"
_TEXT     = "#94a3b8"

PALETTE = {
    "purple":  "#7c3aed",
    "violet":  "#c4b5fd",
    "blue":    "#0ea5e9",
    "yellow":  "#fbbf24",
    "red":     "#f43f5e",
    "green":   "#22c55e",
}


def _style_axes(axes: list) -> None:
    for ax in axes:
        ax.set_facecolor(_BG_PANEL)
        ax.tick_params(colors=_TEXT, labelsize=9)
        for spine in ax.spines.values():
            spine.set_edgecolor(_GRID)
        ax.grid(True, linestyle="--", alpha=0.3, color=_GRID)
        ax.xaxis.label.set_color(_TEXT)
        ax.yaxis.label.set_color(_TEXT)
        ax.title.set_color("white")


def _corner_label(ax, text: str) -> None:
    ax.text(
        0.97, 0.95, text,
        transform=ax.transAxes, ha="right", va="top",
        fontsize=9, color=PALETTE["violet"],
        bbox=dict(facecolor=_BG_DARK, alpha=0.6, edgecolor="none", pad=3),
    )


def _style_table(table, normality_passed: bool, last_row: int) -> None:
    for (row, _), cell in table.get_celld().items():
        cell.set_edgecolor(_GRID)
        if row == 0:
            cell.set_facecolor(_GRID)
            cell.set_text_props(fontweight="bold", color="white")
        elif row == last_row:
            color = "#14532d" if normality_passed else "#7f1d1d"
            cell.set_facecolor(color)
            cell.set_text_props(color="white")
        else:
            cell.set_facecolor(_BG_PANEL if row % 2 == 0 else _BG_DARK)
            cell.set_text_props(color="white")


# ── Main static plot ──────────────────────────────────────────────────────────

def plot_clt(
    dist: Distribution,
    n_samples: int,
    sample_size: int,
    seed: Optional[int] = None,
    save_path: Optional[str] = None,
) -> None:
    """Four-panel CLT dashboard: population, sample means, convergence, stats."""
    rng = np.random.default_rng(seed)
    population = dist.generator(rng, 5000)
    means = dist.sample_means(rng, n_samples, sample_size)
    report = analyse(means, dist, sample_size)

    _print_report(report, dist, n_samples, sample_size)

    fig = plt.figure(figsize=(16, 9), facecolor=_BG_DARK)
    gs = gridspec.GridSpec(
        2, 2, figure=fig,
        hspace=0.45, wspace=0.35,
        left=0.07, right=0.97, top=0.88, bottom=0.08,
    )
    ax_pop   = fig.add_subplot(gs[0, 0])
    ax_means = fig.add_subplot(gs[0, 1])
    ax_conv  = fig.add_subplot(gs[1, 0])
    ax_stats = fig.add_subplot(gs[1, 1])
    ax_stats.axis("off")

    _style_axes([ax_pop, ax_means, ax_conv])

    # ── Panel 1: population distribution ────────────────────────────────────
    ax_pop.hist(
        population, bins=60, density=True,
        color=PALETTE["purple"], alpha=0.75, edgecolor="none",
    )
    x_pop = np.linspace(float(population.min()), float(population.max()), 300)
    kde = stats.gaussian_kde(population)
    ax_pop.plot(x_pop, kde(x_pop), color=PALETTE["violet"], lw=2)
    ax_pop.axvline(dist.theoretical_mean, color=PALETTE["yellow"], lw=1.5, ls="--",
                   label=f"μ = {dist.theoretical_mean:.2f}")
    ax_pop.set_title("Population Distribution", fontsize=13, pad=8)
    ax_pop.set_xlabel("Value", fontsize=11)
    ax_pop.set_ylabel("Density", fontsize=11)
    ax_pop.legend(fontsize=9, framealpha=0.2, facecolor=_BG_PANEL, labelcolor="white")
    _corner_label(ax_pop, dist.label)

    # ── Panel 2: sample means distribution ──────────────────────────────────
    ax_means.hist(
        means, bins=50, density=True,
        color=PALETTE["blue"], alpha=0.75, edgecolor="none",
        label="Sample Means",
    )
    x_fit = np.linspace(float(means.min()), float(means.max()), 300)
    y_empirical = stats.norm.pdf(x_fit, report.empirical_mean, report.empirical_std)
    y_theory    = stats.norm.pdf(x_fit, report.theoretical_mean, report.theoretical_sem)
    ax_means.plot(x_fit, y_empirical, color=PALETTE["red"],    lw=2.5, label="Normal Fit (empirical)")
    ax_means.plot(x_fit, y_theory,    color=PALETTE["yellow"], lw=2.0, ls="--", label="Theoretical CLT")
    ax_means.axvline(report.empirical_mean, color=PALETTE["red"], lw=1.5, ls=":")
    ax_means.set_title(
        f"Sample Means  (n={sample_size:,}, N={n_samples:,})",
        fontsize=13, pad=8,
    )
    ax_means.set_xlabel("Sample Mean", fontsize=11)
    ax_means.set_ylabel("Density", fontsize=11)
    ax_means.legend(fontsize=9, framealpha=0.2, facecolor=_BG_PANEL, labelcolor="white")

    # ── Panel 3: SEM convergence ─────────────────────────────────────────────
    ns = np.unique(np.geomspace(2, max(sample_size, 100), 35).astype(int))
    empirical_sems = []
    for n in ns:
        m = dist.sample_means(rng, min(n_samples, 500), int(n))
        empirical_sems.append(float(np.std(m, ddof=1)))
    theoretical_sems = [dist.theoretical_sem(int(n)) for n in ns]

    ax_conv.plot(ns, theoretical_sems, color=PALETTE["yellow"], lw=2, ls="--",
                 label="Theoretical σ/√n")
    ax_conv.plot(ns, empirical_sems, color=PALETTE["blue"], lw=2,
                 marker="o", ms=4, label="Empirical SEM")
    ax_conv.axvline(sample_size, color=PALETTE["green"], lw=1.5, ls=":",
                    label=f"Current n={sample_size}")
    ax_conv.set_title("Standard Error vs. Sample Size", fontsize=13, pad=8)
    ax_conv.set_xlabel("Sample Size (n)", fontsize=11)
    ax_conv.set_ylabel("Std of Sample Means", fontsize=11)
    ax_conv.set_xscale("log")
    ax_conv.legend(fontsize=9, framealpha=0.2, facecolor=_BG_PANEL, labelcolor="white")

    # ── Panel 4: statistics table ────────────────────────────────────────────
    rows = [
        ["Empirical Mean",  f"{report.empirical_mean:.4f}"],
        ["Theoretical Mean",f"{report.theoretical_mean:.4f}"],
        ["Empirical SEM",   f"{report.empirical_std:.4f}"],
        ["Theoretical SEM", f"{report.theoretical_sem:.4f}"],
        ["KS Statistic",    f"{report.ks_stat:.4f}"],
        ["KS p-value",      f"{report.ks_pvalue:.4f}"],
        ["Shapiro-Wilk W",  f"{report.shapiro_stat:.4f}"],
        ["Shapiro-Wilk p",  f"{report.shapiro_pvalue:.4f}"],
        ["Normality",       "✓ Passed" if report.normality_passed else "✗ Failed"],
    ]
    table = ax_stats.table(
        cellText=rows,
        colLabels=["Statistic", "Value"],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.85)
    _style_table(table, report.normality_passed, last_row=len(rows))

    fig.suptitle(
        "Central Limit Theorem — Interactive Demonstration",
        color="white", fontsize=17, fontweight="bold", y=0.95,
    )

    plt.savefig(save_path, dpi=150, bbox_inches="tight") if save_path else None
    if save_path:
        print(f"Figure saved → {save_path}")

    plt.show()


# ── Convergence animation ─────────────────────────────────────────────────────

def animate_convergence(
    dist: Distribution,
    sample_size: int = 30,
    max_samples: int = 1000,
    seed: Optional[int] = None,
    save_path: Optional[str] = None,
) -> None:
    """Animate how sample means converge to the theoretical normal as N grows."""
    rng = np.random.default_rng(seed)
    all_means = dist.sample_means(rng, max_samples, sample_size)

    fig, ax = plt.subplots(figsize=(10, 6), facecolor=_BG_DARK)
    _style_axes([ax])
    fig.suptitle(
        f"CLT Convergence — {dist.label}  (sample size n={sample_size})",
        color="white", fontsize=14,
    )

    x_fit = np.linspace(float(all_means.min()), float(all_means.max()), 200)
    theory_pdf = stats.norm.pdf(x_fit, dist.theoretical_mean, dist.theoretical_sem(sample_size))
    y_max = float(theory_pdf.max()) * 1.5

    steps = list(range(10, max_samples + 1, max(1, max_samples // 80)))

    def update(frame: int):
        ax.clear()
        _style_axes([ax])
        subset = all_means[:frame]
        ax.hist(subset, bins=min(30, frame), density=True,
                color=PALETTE["blue"], alpha=0.75, edgecolor="none")
        if len(subset) > 5:
            m, s = float(np.mean(subset)), float(np.std(subset, ddof=1))
            ax.plot(x_fit, stats.norm.pdf(x_fit, m, s),
                    color=PALETTE["red"], lw=2, label="Current fit")
        ax.plot(x_fit, theory_pdf, color=PALETTE["yellow"],
                lw=2, ls="--", label="Theoretical CLT")
        ax.set_ylim(0, y_max)
        ax.set_xlabel("Sample Mean", color=_TEXT)
        ax.set_ylabel("Density", color=_TEXT)
        ax.set_title(f"Samples drawn: {frame}", color="white", fontsize=12)
        ax.legend(fontsize=10, framealpha=0.2, facecolor=_BG_PANEL, labelcolor="white")

    ani = FuncAnimation(fig, update, frames=steps, interval=80, repeat=False)

    if save_path:
        ani.save(save_path, fps=15, dpi=120)
        print(f"Animation saved → {save_path}")
    else:
        plt.show()


# ── Console report ────────────────────────────────────────────────────────────

def _print_report(
    report: CLTReport, dist: Distribution, n_samples: int, sample_size: int
) -> None:
    sep = "─" * 46
    print(f"\n{sep}")
    print(f"  Central Limit Theorem — {dist.label}")
    print(f"  N={n_samples:,} samples  ·  n={sample_size} per sample")
    print(sep)
    print(f"  {'Empirical mean':<22} {report.empirical_mean:.4f}")
    print(f"  {'Theoretical mean':<22} {report.theoretical_mean:.4f}")
    print(f"  {'Empirical SEM':<22} {report.empirical_std:.4f}")
    print(f"  {'Theoretical SEM (σ/√n)':<22} {report.theoretical_sem:.4f}")
    print(sep)
    print(f"  {'KS statistic':<22} {report.ks_stat:.4f}  (p={report.ks_pvalue:.4f})")
    print(f"  {'Shapiro-Wilk W':<22} {report.shapiro_stat:.4f}  (p={report.shapiro_pvalue:.4f})")
    status = "PASSED ✓" if report.normality_passed else "FAILED ✗"
    print(f"  {'Normality':<22} {status}")
    print(sep + "\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="plot_sample_means.py",
        description="Central Limit Theorem demonstration across multiple distributions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--distribution", "-d",
        choices=list(DISTRIBUTIONS.keys()),
        default="exponential",
        metavar="DIST",
        help="Population distribution. Choices: " + ", ".join(DISTRIBUTIONS.keys()),
    )
    p.add_argument("--n-samples", "-N", type=int, default=1000,
                   help="Number of samples (sample means) to draw.")
    p.add_argument("--sample-size", "-n", type=int, default=30,
                   help="Observations per sample.")
    p.add_argument("--seed", "-s", type=int, default=None,
                   help="Random seed for reproducibility.")
    p.add_argument("--save", type=str, default=None, metavar="PATH",
                   help="Save static figure to PATH (e.g. output.png).")
    p.add_argument("--animate", action="store_true",
                   help="Show convergence animation instead of static plot.")
    p.add_argument("--save-animation", type=str, default=None, metavar="PATH",
                   help="Save animation to PATH (e.g. output.gif). Requires --animate.")
    p.add_argument("--list-distributions", action="store_true",
                   help="List available distributions and exit.")
    return p


def main(argv: Optional[list] = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.list_distributions:
        print("\nAvailable distributions:\n")
        for key, d in DISTRIBUTIONS.items():
            print(f"  {key:<15}  {d.label}")
        print()
        sys.exit(0)

    dist = DISTRIBUTIONS[args.distribution]

    if args.animate:
        animate_convergence(
            dist=dist,
            sample_size=args.sample_size,
            max_samples=args.n_samples,
            seed=args.seed,
            save_path=args.save_animation,
        )
    else:
        plot_clt(
            dist=dist,
            n_samples=args.n_samples,
            sample_size=args.sample_size,
            seed=args.seed,
            save_path=args.save,
        )


if __name__ == "__main__":
    main()
