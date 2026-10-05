"""Charts that show the evidence behind each flag."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .detectors import DETECTORS, MEASURES

_SUSPECT, _NORMAL = "#c0392b", "#7f8c8d"


def _top_site(result, detector):
    return result[detector].idxmin()


def plot_heatmap(result: pd.DataFrame, path: Path, truth: pd.DataFrame | None = None) -> None:
    cols = list(DETECTORS)
    table = result.sort_index()
    scores = np.minimum(-np.log10(table[cols].to_numpy()), 20)
    fig, ax = plt.subplots(figsize=(7, 0.36 * len(table) + 1.6))
    im = ax.imshow(scores, cmap="Reds", aspect="auto", vmin=0, vmax=20)
    labels = []
    for site in table.index:
        label = site
        if truth is not None:
            kind = truth.set_index("site_id").loc[site, "fraud_type"]
            if kind:
                label = f"{site}  (truth: {kind})"
        labels.append(label)
    ax.set_yticks(range(len(table)), labels)
    ax.set_xticks(range(len(cols)), [c.replace("_", "\n") for c in cols])
    for i, site in enumerate(table.index):
        fired = table.loc[site, "reasons"].split(", ")
        for j, col in enumerate(cols):
            if col in fired:
                ax.text(j, i, "FLAG", ha="center", va="center", color="white", fontsize=8,
                        fontweight="bold")
    fig.colorbar(im, ax=ax, label="evidence: -log10(p), capped at 20")
    ax.set_title("Which sites look suspicious, and why")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_digits(df: pd.DataFrame, result: pd.DataFrame, path: Path) -> None:
    site = _top_site(result, "digit_preference")
    is_site = df["site_id"] == site

    def share(rows):
        d = pd.concat([rows[m].abs().astype(int) % 10 for m in MEASURES])
        return d.value_counts(normalize=True).reindex(range(10), fill_value=0)

    fig, ax = plt.subplots(figsize=(7, 3.8))
    x = np.arange(10)
    ax.bar(x - 0.2, share(df[~is_site]), 0.4, label="all other sites", color=_NORMAL)
    ax.bar(x + 0.2, share(df[is_site]), 0.4, label=f"site {site}", color=_SUSPECT)
    ax.set_xticks(x)
    ax.set_xlabel("last digit of the recorded value")
    ax.set_ylabel("share of values")
    ax.set_title("Last digits: invented numbers favour some endings")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_variability(df: pd.DataFrame, result: pd.DataFrame, path: Path) -> None:
    resid = df["sbp"] - df.groupby(["arm", "visit_week"])["sbp"].transform("mean")
    sd = resid.groupby(df["patient_id"]).std()
    site_of = df.drop_duplicates("patient_id").set_index("patient_id")["site_id"]
    sites = sorted(site_of.unique())
    flagged = {s for s in sites if "variability" in result.loc[s, "reasons"].split(", ")}
    fig, ax = plt.subplots(figsize=(9, 3.8))
    boxes = ax.boxplot([sd[site_of == s] for s in sites], patch_artist=True, showfliers=False)
    for box, s in zip(boxes["boxes"], sites):
        box.set_facecolor(_SUSPECT if s in flagged else "#d5d8dc")
    ax.set_xticks(range(1, len(sites) + 1), sites, rotation=60)
    ax.set_ylabel("visit-to-visit variation per patient\n(systolic BP, mmHg)")
    ax.set_title("Natural variation by site (red = flagged by the variability check)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_correlation(df: pd.DataFrame, result: pd.DataFrame, path: Path) -> None:
    site = _top_site(result, "correlation")
    base = df[df["visit_week"] == df["visit_week"].min()]
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.8), sharex=True, sharey=True)
    for ax, rows, title, color in [
        (axes[0], base[base["site_id"] != site], "all other sites", _NORMAL),
        (axes[1], base[base["site_id"] == site], f"site {site}", _SUSPECT),
    ]:
        ax.scatter(rows["sbp"], rows["dbp"], s=10, alpha=0.5, color=color)
        r = rows["sbp"].corr(rows["dbp"])
        ax.set_title(f"{title} (correlation {r:.2f})")
        ax.set_xlabel("systolic BP at first visit")
    axes[0].set_ylabel("diastolic BP at first visit")
    fig.suptitle("Real measurements move together; invented ones often do not")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def make_all(df, result, out_dir: Path, truth=None) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [out_dir / f"{n}.png" for n in ("heatmap", "digits", "variability", "correlation")]
    plot_heatmap(result, paths[0], truth)
    plot_digits(df, result, paths[1])
    plot_variability(df, result, paths[2])
    plot_correlation(df, result, paths[3])
    return paths
