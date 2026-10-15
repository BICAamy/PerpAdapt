"""Evaluate corrected PPC, Lee(4), and Lee(6) on one dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from PC import PC
from PPC import PPC


def load_dataset(path, limit=None):
    with np.load(path, allow_pickle=False) as archive:
        dataset = {key: archive[key] for key in archive.files}
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be positive")
        instance_count = len(dataset["t"])
        for key, value in list(dataset.items()):
            if value.ndim > 0 and value.shape[0] == instance_count:
                dataset[key] = value[:limit]
    return dataset


def evaluate(dataset):
    return [
        PPC(sac=None).calibrate(dataset, verbose=False),
        PC(sac=None, pc_circle_num=4).calibrate(dataset, verbose=False),
        PC(sac=None, pc_circle_num=6).calibrate(dataset, verbose=False),
    ]


def _format(value):
    if value is None or not np.isfinite(value):
        return "n/a"
    return f"{value:.12e}"


def print_results(dataset, results):
    instance_count = len(dataset["t"])
    samples = int(np.asarray(dataset.get("samples_per_circle", 100)).item())
    noise_std = float(np.asarray(dataset.get("noise_std", np.nan)).item())
    print(
        f"instances={instance_count}, samples_per_circle={samples}, "
        f"noise_std={noise_std:g} m"
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
    parser.add_argument(
        "--limit", type=int, default=None, help="optionally evaluate only the first N instances"
    )
    parser.add_argument(
        "--json", dest="json_path", default=None, help="optional path for metric JSON"
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    dataset = load_dataset(args.dataset, args.limit)
    results = evaluate(dataset)
    print_results(dataset, results)

    if args.json_path:
        serializable = []
        for result in results:
            serializable.append(
                {
                    "method": result["method"],
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
