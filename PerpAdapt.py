"""Command-line entry point for corrected PerpAdapt calibration.

Examples::

    python PerpAdapt.py --dataset data.npz --method PPC --sac None
    python PerpAdapt.py --dataset data.npz --method PPC --sac RANSAC
    python PerpAdapt.py --dataset data.npz --method PPC --sac MemorySAC
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from geometry import DEFAULT_MIN_TRIANGLE_QUALITY
from PC import PC
from PPC import PPC


def _casefold_choice(value: str) -> str:
    return value.casefold()


def add_sac_arguments(parser):
    """Add the shared None/RANSAC/MemorySAC command-line options."""
    parser.add_argument(
        "--sac",
        type=_casefold_choice,
        choices=("none", "ransac", "memorysac"),
        default="none",
        help="inlier selector: None, RANSAC, or MemorySAC (case-insensitive)",
    )
    consensus = parser.add_argument_group("RANSAC / MemorySAC")
    consensus.add_argument("--num-iterations", type=int, default=10)
    consensus.add_argument(
        "--distance-threshold",
        type=float,
        default=0.05,
        help="point-to-circle inlier threshold in the dataset's length unit",
    )
    consensus.add_argument(
        "--min-triangle-quality",
        type=float,
        default=DEFAULT_MIN_TRIANGLE_QUALITY,
        help=(
            "reject a sampled triplet when ||e01 x e02|| / longest_edge^2 "
            "is below this value"
        ),
    )
    consensus.add_argument(
        "--max-sampling-attempts",
        type=int,
        default=1000,
        help="maximum total draws used to obtain all valid hypotheses",
    )
    consensus.add_argument(
        "--sac-seed",
        type=int,
        default=20261017,
        help="random seed for RANSAC/MemorySAC triplet draws",
    )
    consensus.add_argument(
        "--update-rate",
        type=float,
        default=0.001,
        help="MemorySAC score-update rate",
    )
    consensus.add_argument(
        "--checkpoint",
        default="net.pth",
        help="MemorySAC network checkpoint",
    )
    return parser


def load_dataset(path, *, start=0, limit=None, samples_per_circle=None):
    """Load an NPZ file and optionally select a contiguous instance range."""
    if start < 0:
        raise ValueError("start must be non-negative")
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")

    with np.load(path, allow_pickle=False) as archive:
        dataset = {key: archive[key] for key in archive.files}
    if "t" not in dataset:
        raise KeyError("dataset is missing required field 't'")

    instance_count = len(dataset["t"])
    stop = instance_count if limit is None else min(instance_count, start + limit)
    if start >= instance_count:
        raise ValueError(
            f"start={start} is outside a dataset with {instance_count} instances"
        )
    for key, value in list(dataset.items()):
        if value.ndim > 0 and value.shape[0] == instance_count:
            dataset[key] = value[start:stop]

    if samples_per_circle is not None:
        if samples_per_circle < 3:
            raise ValueError("samples_per_circle must be at least 3")
        dataset["samples_per_circle"] = np.array(
            samples_per_circle, dtype=np.int64
        )
    elif "samples_per_circle" not in dataset:
        alpha_positions = np.asarray(dataset["alpha_t"])
        if alpha_positions.ndim == 4:
            inferred = alpha_positions.shape[2]
        elif alpha_positions.ndim == 3 and alpha_positions.shape[1] % 3 == 0:
            inferred = alpha_positions.shape[1] // 3
        else:
            raise ValueError(
                "cannot infer samples per circle; pass --samples-per-circle"
            )
        dataset["samples_per_circle"] = np.array(inferred, dtype=np.int64)
    return dataset


def build_sac(args):
    """Construct the selected sampler; heavy MemorySAC imports stay optional."""
    common = {
        "num_iterations": args.num_iterations,
        "sample_size": 3,
        "distance_threshold": args.distance_threshold,
        "min_triangle_quality": args.min_triangle_quality,
        "max_sampling_attempts": args.max_sampling_attempts,
        "random_state": args.sac_seed,
    }
    if args.sac == "none":
        return None
    if args.sac == "ransac":
        from RANSAC import RANSAC

        return RANSAC(**common)

    from MemorySAC import MemorySAC

    return MemorySAC(
        **common,
        update_rate=args.update_rate,
        pth=args.checkpoint,
    )


def build_calibrator(method, sac):
    if method == "ppc":
        return PPC(sac=sac)
    if method == "lee4":
        return PC(sac=sac, pc_circle_num=4)
    return PC(sac=sac, pc_circle_num=6)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run corrected PPC or Lee PC with a selectable consensus method."
    )
    parser.add_argument("--dataset", default="dataset.npz", help="input NPZ file")
    parser.add_argument(
        "--method",
        type=_casefold_choice,
        choices=("ppc", "lee4", "lee6"),
        default="ppc",
        help="calibration method (case-insensitive)",
    )
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument(
        "--limit", type=int, default=None, help="run only this many instances"
    )
    parser.add_argument(
        "--samples-per-circle",
        type=int,
        default=None,
        help="override/inject per-trajectory frame count for a legacy dataset",
    )

    return add_sac_arguments(parser)


def main(argv=None):
    args = build_parser().parse_args(argv)
    dataset_path = Path(args.dataset)
    dataset = load_dataset(
        dataset_path,
        start=args.start,
        limit=args.limit,
        samples_per_circle=args.samples_per_circle,
    )
    sac = build_sac(args)
    calibrator = build_calibrator(args.method, sac)

    print(
        f"dataset={dataset_path}, instances={len(dataset['t'])}, "
        f"method={args.method}, sac={args.sac}"
    )
    if sac is not None:
        print(
            f"sac_seed={args.sac_seed}, iterations={args.num_iterations}, "
            f"min_triangle_quality={args.min_triangle_quality:g}"
        )
    calibrator.calibrate(dataset)


if __name__ == "__main__":
    main()
