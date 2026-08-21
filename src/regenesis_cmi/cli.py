"""Command-line entry points for validation, experiments, and reporting."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .evaluation.benchmark import (
    estimate_runtime,
    load_config,
    regenerate_report,
    run_benchmark,
    validate_exact_worlds,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="regenesis-cmi",
        description="Synthetic validation for Regenesis conditional information measurements",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    exact = subparsers.add_parser("validate-exact", help="run analytic Generator A checks")
    exact.add_argument("--output", type=Path, help="optional directory for validation CSVs")

    run = subparsers.add_parser("run", help="run a configured synthetic benchmark")
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--output", type=Path)
    run.add_argument(
        "--core",
        action="store_true",
        help="run estimator core only; full smoke is required for control figures",
    )

    report = subparsers.add_parser("report", help="regenerate all figures from saved tables")
    report.add_argument("--results", type=Path, required=True)

    runtime = subparsers.add_parser(
        "estimate-runtime", help="estimate configured CPU runtime without running it"
    )
    runtime.add_argument("--config", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "validate-exact":
        entropy, exact, gaussian = validate_exact_worlds()
        if args.output:
            args.output.mkdir(parents=True, exist_ok=True)
            entropy.to_csv(args.output / "entropy_sanity.csv", index=False)
            exact.to_csv(args.output / "exact_validation.csv", index=False)
            gaussian.to_csv(args.output / "gaussian_validation.csv", index=False)
        print(exact[["world", "direction", "oracle_cmi_bits", "estimate_bits", "bias_bits"]].to_string(index=False))
        print("\nGaussian benchmark")
        print(gaussian.to_string(index=False))
        passed = (exact["absolute_error_bits"] < 1e-10).all()
        passed = passed and (gaussian["absolute_error_bits"] < 0.01).all()
        return 0 if passed else 1
    if args.command == "run":
        output = run_benchmark(
            args.config, output_override=args.output, core_only=args.core
        )
        metadata = json.loads((output / "run_metadata.json").read_text(encoding="utf-8"))
        print(json.dumps({"output": str(output), **metadata}, indent=2))
        return 0 if metadata["scientific_checks_passed"] else 2
    if args.command == "report":
        output = regenerate_report(args.results)
        print(output)
        return 0
    if args.command == "estimate-runtime":
        print(json.dumps(estimate_runtime(load_config(args.config)), indent=2))
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

