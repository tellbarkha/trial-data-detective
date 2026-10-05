"""Four statistical checks, each comparing one site against all the others.

Every detector takes the long-format trial table and returns a pandas Series
of p-values indexed by site. A small p-value means "this site looks unlike
the rest of the trial in a way that honest data rarely does".
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import cdist

MEASURES = ("sbp", "dbp", "hr")
REQUIRED_COLUMNS = ("site_id", "patient_id", "arm", "visit_week", *MEASURES)


def validate(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Data is missing required columns: {missing}")
    if df["site_id"].nunique() < 3:
        raise ValueError("Need at least 3 sites to compare sites against each other")


def _patient_site(df):
    return df.drop_duplicates("patient_id").set_index("patient_id")["site_id"]


def _wide(df, measure):
    return df.pivot(index="patient_id", columns="visit_week", values=measure)


def digit_preference(df: pd.DataFrame) -> pd.Series:
    """Last-digit test.

    In honest measurements the last digit is close to random. People who make
    numbers up favour some digits. Chi-square test of each site's last-digit
    counts against the other sites (not against a perfectly even spread,
    because honest staff also round a little).

    Honest sites differ in how much they round, which inflates every site's
    chi-square statistic. We estimate that inflation from the median site
    (the "genomic control" trick from genetics) and divide it out, so only
    sites that stand out from normal site-to-site differences get small
    p-values.
    """
    digits = pd.concat([df[m].abs().astype(int) % 10 for m in MEASURES], ignore_index=True)
    sites = pd.concat([df["site_id"]] * len(MEASURES), ignore_index=True)
    table = pd.crosstab(sites, digits).reindex(columns=range(10), fill_value=0)
    table = table.loc[:, table.sum() > 0]
    total = table.sum()
    dof = table.shape[1] - 1
    chi2 = pd.Series(
        {
            site: stats.chi2_contingency(np.vstack([row, total - row]))[0]
            for site, row in table.iterrows()
        }
    )
    inflation = max(1.0, chi2.median() / stats.chi2.median(dof))
    return pd.Series(stats.chi2.sf(chi2 / inflation, dof), index=chi2.index, name="digit_preference")


def variability(df: pd.DataFrame) -> pd.Series:
    """Too-tidy (or too-messy) test.

    Checks two kinds of spread: between patients at the first visit
    (Brown-Forsythe test) and within each patient across visits (Mann-Whitney
    test on per-patient standard deviations, after removing the average
    treatment trend). The smallest of the four p-values is Bonferroni-adjusted.
    """
    site_of = _patient_site(df)
    first_week = df["visit_week"].min()
    baseline = df[df["visit_week"] == first_week].set_index("patient_id")
    within = {}
    for m in ("sbp", "hr"):
        resid = df[m] - df.groupby(["arm", "visit_week"])[m].transform("mean")
        within[m] = resid.groupby(df["patient_id"]).std()

    out = {}
    for site in site_of.unique():
        ids = site_of.index[site_of == site]
        others = site_of.index[site_of != site]
        pvals = [
            stats.levene(baseline.loc[ids, m], baseline.loc[others, m], center="median")[1]
            for m in ("sbp", "dbp")
        ]
        pvals += [
            stats.mannwhitneyu(w.loc[ids].dropna(), w.loc[others].dropna())[1]
            for w in within.values()
        ]
        out[site] = min(1.0, min(pvals) * len(pvals))
    return pd.Series(out, name="variability")


def _closeness(matrix: np.ndarray) -> float:
    """10th percentile of each patient's distance to their nearest neighbour."""
    d = cdist(matrix, matrix)
    np.fill_diagonal(d, np.inf)
    return float(np.quantile(d.min(axis=1), 0.10))


def duplicates(df: pd.DataFrame, n_reference: int = 200, seed: int = 0) -> pd.Series:
    """Copy-paste test.

    Each patient is a point made of all their measurements. For every site we
    ask how close the closest pairs of patients are, then compare with random
    groups of the same size drawn from the other sites. The p-value uses a
    normal approximation (on the log scale) to that reference distribution,
    so it is approximate.
    """
    rng = np.random.default_rng(seed)
    site_of = _patient_site(df)
    wide = pd.concat([_wide(df, m) for m in MEASURES], axis=1).dropna()
    site_of = site_of.loc[wide.index]
    x = wide.to_numpy(dtype=float)
    x = x / x.std(axis=0)
    eps = 1e-3  # avoids log(0) when patients are exact copies

    out = {}
    for site in site_of.unique():
        mask = (site_of == site).to_numpy()
        own, pool = x[mask], x[~mask]
        n = min(len(own), len(pool))
        if n < 5:
            out[site] = 1.0
            continue
        observed = np.log(_closeness(own) + eps)
        ref = np.array(
            [
                np.log(_closeness(pool[rng.choice(len(pool), n, replace=False)]) + eps)
                for _ in range(n_reference)
            ]
        )
        out[site] = float(stats.norm.cdf((observed - ref.mean()) / ref.std(ddof=1)))
    return pd.Series(out, name="duplicates")


def correlation(df: pd.DataFrame) -> pd.Series:
    """Missing-relationship test.

    Real measurements are linked: systolic and diastolic pressure move
    together, and a patient's last visit resembles their first. Invented
    numbers often lack these links. For both pairs, each site's correlation
    (on the Fisher z scale) is compared with the median of the other sites,
    which a few fraudulent sites cannot drag around. Bonferroni-adjusted.
    """
    site_of = _patient_site(df)
    sbp, dbp = _wide(df, "sbp"), _wide(df, "dbp")
    first, last = sbp.columns.min(), sbp.columns.max()
    pairs = [(sbp[first], dbp[first]), (sbp[first], sbp[last])]

    pvals = []
    for a, b in pairs:
        both = pd.concat([a, b], axis=1, keys=["a", "b"]).dropna()
        groups = both.groupby(site_of.loc[both.index])
        n = groups.size()
        r = groups.apply(lambda g: g["a"].corr(g["b"])).clip(-0.999999, 0.999999)
        ok = n >= 5
        z, var = np.arctanh(r[ok]), 1 / (n[ok] - 3)
        p = pd.Series(1.0, index=n.index)
        for site in z.index:
            others = z.index != site
            # variance of a median of k roughly-normal values is about (pi/2) * var / k
            ref_var = (np.pi / 2) * var[others].mean() / others.sum()
            score = (z[site] - z[others].median()) / np.sqrt(var[site] + ref_var)
            p[site] = 2 * stats.norm.sf(abs(score))
        pvals.append(p)
    combined = pd.concat(pvals, axis=1).min(axis=1) * len(pairs)
    return combined.clip(upper=1.0).rename("correlation")


DETECTORS = {
    "digit_preference": digit_preference,
    "variability": variability,
    "duplicates": duplicates,
    "correlation": correlation,
}
