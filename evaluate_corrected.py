"""Evaluate corrected PPC, Lee(4), and Lee(6) on one dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from PerpAdapt import add_sac_arguments, build_sac, load_dataset
from PC import PC
from PPC import PPC


def evaluate(dataset, sac_factory=None):
    """Evaluate all three methods with fresh, equivalently configured SACs.

    A new sampler is created for each calibration method so mutable random and
    memory state consumed by PPC cannot leak into Lee(4) or Lee(6).
    """
    if sac_factory is None:
        def sac_factory():
            return None

    calibrator_factories = (
        lambda sac: PPC(sac=sac),
        lambda sac: PC(sac=sac, pc_circle_num=4),
        lambda sac: PC(sac=sac, pc_circle_num=6),
    )
    results = []
    for make_calibrator in calibrator_factories:
        calibrator = make_calibrator(sac_factory())
        results.append(calibrator.calibrate(dataset, verbose=False))
    return results


def _format(value):
    if value is None or not np.isfinite(value):
        return "n/a"
    return f"{value:.12e}"


def print_results(dataset, results, sac_name="none"):
    instance_count = len(dataset["t"])
    samples = int(np.asarray(dataset.get("samples_per_circle", 100)).item())
    noise_std = float(np.asarray(dataset.get("noise_std", np.nan)).item())
    print(
        f"instances={instance_count}, samples_per_circle={samples}, "
        f"noise_std={noise_std:g} m, sac={sac_name}"
    )
    columns = [
        ("method", 10),
        ("c_c,0 mean error", 21),
        ("t component MAE", 21),
        ("t component MSE", 21),
        ("R^2", 17),
        ("t mean L2", 21),
    ]
    print("  ".join(name.ljust(width) for name, width in columns))
    print("  ".join("-" * width for _, width in columns))
    for result in results:
        values = [
            result["method"],
            _format(result.get("cc_error_mean")),
            _format(result["mae"]),
            _format(result["mse"]),
            _format(result["r2"]),
            _format(result["t_l2_mean"]),
        ]
        print(
            "  ".join(
                value.ljust(width) for value, (_, width) in zip(values, columns)
            )
        )


def build_parser():
    parser = argparse.ArgumentParser(
        description="Compare corrected PPC with corrected Lee PC(4)/(6)."
    )
    parser.add_argument("--dataset", required=True, help="path to a .npz dataset")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="optionally evaluate only the first N instances",
    )
    parser.add_argument(
        "--samples-per-circle",
        type=int,
        default=None,
        help="override/inject per-trajectory frame count for a legacy dataset",
    )
    parser.add_argument(
        "--json",
        dest="json_path",
        default=None,
        help="optional path for metric JSON",
    )
    return add_sac_arguments(parser)


def main(argv=None):
    args = build_parser().parse_args(argv)
    dataset = load_dataset(
        args.dataset,
        start=args.start,
        limit=args.limit,
        samples_per_circle=args.samples_per_circle,
    )
    results = evaluate(dataset, sac_factory=lambda: build_sac(args))
    print_results(dataset, results, sac_name=args.sac)

    if args.json_path:
        serializable = []
        for result in results:
            serializable.append(
                {
                    "method": result["method"],
                    "sac": args.sac,
                    "sac_seed": args.sac_seed if args.sac != "none" else None,
                    "cc_error_mean": result.get("cc_error_mean"),
                    "t_component_mae": result["mae"],
                    "t_component_mse": result["mse"],
                    "r2": result["r2"],
                    "t_l2_mean": result["t_l2_mean"],
                }
            )
        output = Path(args.json_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(serializable, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"saved metrics: {output}")


if __name__ == "__main__":
    main()
