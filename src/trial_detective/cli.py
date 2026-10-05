"""Command line interface: simulate, detect, benchmark."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from . import plots
from .scoring import benchmark, evaluate, run_all
from .simulate import TrialConfig, simulate_trial


def _write_report(result, out_dir: Path, score: dict | None) -> Path:
    lines = ["# Site screening report", ""]
    flagged = result[result["flagged"]]
    lines.append(f"{len(flagged)} of {len(result)} sites flagged for review.")
    lines.append("")
    lines.append("| site | flagged | reasons | suspicion |")
    lines.append("|---|---|---|---|")
    for site, row in result.iterrows():
        mark = "YES" if row["flagged"] else ""
        lines.append(f"| {site} | {mark} | {row['reasons']} | {row['suspicion']:.1f} |")
    if score:
        lines += ["", "## Compared with the known answer", ""]
        lines += [f"- {k.replace('_', ' ')}: {v:.2f}" if isinstance(v, float)
                  else f"- {k.replace('_', ' ')}: {v}" for k, v in score.items()]
    lines += ["", "A flag means a site deserves a closer look. It is not proof of misconduct.", ""]
    for name in ("heatmap", "digits", "variability", "correlation"):
        lines.append(f"![{name}]({name}.png)")
    path = out_dir / "report.md"
    path.write_text("\n".join(lines) + "\n")
    return path


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="trial-detective", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("simulate", help="create a pretend trial with some cheating sites")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sites", type=int, default=20)
    p.add_argument("--out", type=Path, default=Path("output"))

    p = sub.add_parser("detect", help="screen a trial CSV and write a report")
    p.add_argument("data", type=Path)
    p.add_argument("--truth", type=Path, help="optional answer key to score against")
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--out", type=Path, default=Path("output/report"))

    p = sub.add_parser("benchmark", help="measure accuracy over many simulated trials")
    p.add_argument("--trials", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)

    args = parser.parse_args(argv)

    if args.command == "simulate":
        df, truth = simulate_trial(TrialConfig(n_sites=args.sites, seed=args.seed))
        args.out.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.out / "trial.csv", index=False)
        truth.to_csv(args.out / "truth.csv", index=False)
        print(f"Wrote {len(df)} rows for {df['patient_id'].nunique()} patients to {args.out}/")

    elif args.command == "detect":
        df = pd.read_csv(args.data)
        truth = pd.read_csv(args.truth, keep_default_na=False) if args.truth else None
        result = run_all(df, args.alpha)
        args.out.mkdir(parents=True, exist_ok=True)
        result.to_csv(args.out / "site_scores.csv")
        plots.make_all(df, result, args.out, truth)
        score = evaluate(result, truth) if truth is not None else None
        report = _write_report(result, args.out, score)
        print(result[result["flagged"]][["suspicion", "reasons"]].round(1).to_string())
        if score:
            print(f"\nrecall {score['recall']:.2f}, precision {score['precision']:.2f}")
        print(f"\nReport: {report}")

    elif args.command == "benchmark":
        summary = benchmark(args.trials, args.seed)
        print(summary.round(3).to_string())


if __name__ == "__main__":
    main()
