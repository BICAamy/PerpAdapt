"""Generate reproducible PerpAdapt simulation datasets.

The fixed base angles are specified in degrees and converted exactly once to
radians. Gaussian noise and structured outlier blocks are added only to the
measured world translations; rotations and clean ground truth stay unchanged.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


ALPHA_BASE_DEG = np.array([0.0, 90.0, -90.0], dtype=np.float64)
BETA_BASE_DEG = np.array([0.0, 45.0, -45.0], dtype=np.float64)
NOISE_MODES = ("none", "gaussian", "structured", "both")


def rotation_x(angle):
    cosine, sine = np.cos(angle), np.sin(angle)
    return np.array(
        [[1.0, 0.0, 0.0], [0.0, cosine, -sine], [0.0, sine, cosine]],
        dtype=np.float64,
    )


def rotation_y(angle):
    cosine, sine = np.cos(angle), np.sin(angle)
    return np.array(
        [[cosine, 0.0, sine], [0.0, 1.0, 0.0], [-sine, 0.0, cosine]],
        dtype=np.float64,
    )


def _random_unit_vector(rng):
    while True:
        direction = rng.normal(size=3)
        norm = np.linalg.norm(direction)
        if norm > np.finfo(np.float64).eps:
            return direction / norm


def _add_structured_outliers(
    points,
    samples_per_circle,
    rng,
    outlier_fraction,
    outlier_offset,
):
    """Shift one contiguous circular block in each of three trajectories."""
    points = np.asarray(points, dtype=np.float64).copy()
    mask = np.zeros(len(points), dtype=bool)
    offsets = np.zeros((3, 3), dtype=np.float64)
    starts = np.full(3, -1, dtype=np.int64)

    if outlier_fraction == 0.0 or outlier_offset == 0.0:
        return points, mask, offsets, starts, 0

    outlier_count = min(
        samples_per_circle,
        max(1, int(round(samples_per_circle * outlier_fraction))),
    )
    local_sequence = np.arange(outlier_count, dtype=np.int64)
    for circle_index in range(3):
        start = int(rng.integers(0, samples_per_circle))
        local_indices = (start + local_sequence) % samples_per_circle
        indices = circle_index * samples_per_circle + local_indices
        offset = outlier_offset * _random_unit_vector(rng)
        points[indices] += offset
        mask[indices] = True
        offsets[circle_index] = offset
        starts[circle_index] = start
    return points, mask, offsets, starts, outlier_count


def generate_dataset(
    instances=1000,
    samples_per_circle=100,
    seed=20261015,
    noise_std=0.0,
    noise_seed=20261016,
    noise_mode="gaussian",
    outlier_fraction=0.1,
    outlier_offset=0.03,
    outlier_seed=20261017,
):
    """Generate arrays compatible with the original ``dataset.npz`` layout.

    The coordinate unit is metres.  For example, ``noise_std=0.001`` means
    independent 1 mm standard-deviation Gaussian noise in every translation
    component of every pose. Structured outliers are one contiguous block per
    trajectory whose points share one fixed-magnitude 3-D offset.
    """
    if instances < 1:
        raise ValueError("instances must be positive")
    if samples_per_circle < 3:
        raise ValueError("samples_per_circle must be at least 3")
    if noise_std < 0.0:
        raise ValueError("noise_std cannot be negative")
    noise_mode = str(noise_mode).casefold()
    if noise_mode not in NOISE_MODES:
        raise ValueError(f"noise_mode must be one of {NOISE_MODES}")
    if not 0.0 <= outlier_fraction <= 1.0:
        raise ValueError("outlier_fraction must be in [0, 1]")
    if outlier_offset < 0.0:
        raise ValueError("outlier_offset cannot be negative")

    use_gaussian = noise_mode in ("gaussian", "both")
    use_structured = noise_mode in ("structured", "both")
    applied_noise_std = float(noise_std) if use_gaussian else 0.0
    applied_outlier_fraction = float(outlier_fraction) if use_structured else 0.0
    applied_outlier_offset = float(outlier_offset) if use_structured else 0.0

    data_rng = np.random.default_rng(seed)
    noise_rng = np.random.default_rng(noise_seed)
    outlier_rng = np.random.default_rng(outlier_seed)
    alpha_base = np.deg2rad(ALPHA_BASE_DEG)
    beta_base = np.deg2rad(BETA_BASE_DEG)
    sweep_angles = np.linspace(
        0.0, 2.0 * np.pi, samples_per_circle, dtype=np.float64
    )

    alpha_quaternions = np.empty(
        (instances, 3 * samples_per_circle, 4), dtype=np.float64
    )
    alpha_positions = np.empty(
        (instances, 3 * samples_per_circle, 3), dtype=np.float64
    )
    beta_quaternions = np.empty_like(alpha_quaternions)
    beta_positions = np.empty_like(alpha_positions)
    labels = np.empty((instances, 3), dtype=np.float64)
    pivots = np.empty((instances, 3, 3), dtype=np.float64)
    alpha_outlier_masks = np.zeros(
        (instances, 3 * samples_per_circle), dtype=bool
    )
    beta_outlier_masks = np.zeros_like(alpha_outlier_masks)
    alpha_outlier_offsets = np.zeros((instances, 3, 3), dtype=np.float64)
    beta_outlier_offsets = np.zeros_like(alpha_outlier_offsets)
    alpha_outlier_starts = np.full((instances, 3), -1, dtype=np.int64)
    beta_outlier_starts = np.full_like(alpha_outlier_starts, -1)
    outlier_count_per_circle = 0

    for instance in range(instances):
        tx, ty, tz = data_rng.uniform(0.0, 1.0, size=3)
        minor_radius = data_rng.uniform(0.6, 1.0)
        major_radius = data_rng.uniform(0.0, 0.2)
        sensor_offset = np.array(
            [tx, ty, minor_radius + tz], dtype=np.float64
        )
        arm_offset = np.array([0.0, 0.0, major_radius], dtype=np.float64)
        sensor_mount_rotation = Rotation.random(random_state=data_rng).as_matrix()

        alpha_matrices = []
        alpha_points = []
        for alpha in alpha_base:
            rotation_alpha = rotation_x(alpha)
            for beta in sweep_angles:
                moving_rotation = rotation_alpha @ rotation_y(beta)
                alpha_matrices.append(moving_rotation @ sensor_mount_rotation)
                alpha_points.append(
                    moving_rotation @ sensor_offset + rotation_alpha @ arm_offset
                )

        beta_matrices = []
        beta_points = []
        for beta in beta_base:
            rotation_beta = rotation_y(beta)
            for alpha in sweep_angles:
                rotation_alpha = rotation_x(alpha)
                moving_rotation = rotation_alpha @ rotation_beta
                beta_matrices.append(moving_rotation @ sensor_mount_rotation)
                beta_points.append(
                    moving_rotation @ sensor_offset + rotation_alpha @ arm_offset
                )

        alpha_points = np.asarray(alpha_points, dtype=np.float64)
        beta_points = np.asarray(beta_points, dtype=np.float64)
        if applied_noise_std > 0.0:
            # Keep this per-instance alpha-then-beta order so a fixed noise seed
            # reproduces the documented experiments exactly.
            alpha_points = alpha_points + noise_rng.normal(
                0.0, applied_noise_std, size=alpha_points.shape
            )
            beta_points = beta_points + noise_rng.normal(
                0.0, applied_noise_std, size=beta_points.shape
            )

        if use_structured:
            (
                alpha_points,
                alpha_outlier_masks[instance],
                alpha_outlier_offsets[instance],
                alpha_outlier_starts[instance],
                outlier_count_per_circle,
            ) = _add_structured_outliers(
                alpha_points,
                samples_per_circle,
                outlier_rng,
                applied_outlier_fraction,
                applied_outlier_offset,
            )
            (
                beta_points,
                beta_outlier_masks[instance],
                beta_outlier_offsets[instance],
                beta_outlier_starts[instance],
                outlier_count_per_circle,
            ) = _add_structured_outliers(
                beta_points,
                samples_per_circle,
                outlier_rng,
                applied_outlier_fraction,
                applied_outlier_offset,
            )

        alpha_quaternions[instance] = Rotation.from_matrix(
            np.asarray(alpha_matrices)
        ).as_quat()
        beta_quaternions[instance] = Rotation.from_matrix(
            np.asarray(beta_matrices)
        ).as_quat()
        alpha_positions[instance] = alpha_points
        beta_positions[instance] = beta_points

        # T_world_sensor = [R2 R1, R2 p + A q].  Therefore the same physical
        # pivot expressed in the sensor frame is -R1^T p for every pose.
        labels[instance] = -sensor_mount_rotation.T @ sensor_offset
        pivots[instance] = np.asarray(
            [rotation_x(alpha) @ arm_offset for alpha in alpha_base]
        )

    return {
        "alpha_R": alpha_quaternions,
        "alpha_t": alpha_positions,
        "beta_R": beta_quaternions,
        "beta_t": beta_positions,
        "t": labels,
        "cc": pivots[:, 0],
        "cc_all": pivots,
        "alpha_outlier_mask": alpha_outlier_masks,
        "beta_outlier_mask": beta_outlier_masks,
        "alpha_outlier_offsets": alpha_outlier_offsets,
        "beta_outlier_offsets": beta_outlier_offsets,
        "alpha_outlier_starts": alpha_outlier_starts,
        "beta_outlier_starts": beta_outlier_starts,
        "samples_per_circle": np.array(samples_per_circle, dtype=np.int64),
        "alpha_base_deg": ALPHA_BASE_DEG.copy(),
        "beta_base_deg": BETA_BASE_DEG.copy(),
        "seed": np.array(seed, dtype=np.int64),
        "noise_seed": np.array(noise_seed, dtype=np.int64),
        "outlier_seed": np.array(outlier_seed, dtype=np.int64),
        "noise_mode": np.array(noise_mode),
        "noise_std": np.array(applied_noise_std, dtype=np.float64),
        "outlier_fraction": np.array(
            applied_outlier_fraction, dtype=np.float64
        ),
        "outlier_offset": np.array(applied_outlier_offset, dtype=np.float64),
        "outlier_count_per_circle": np.array(
            outlier_count_per_circle, dtype=np.int64
        ),
        "position_unit": np.array("m"),
        "format_version": np.array("perpadapt-sim-v3"),
    }


def build_parser():
    parser = argparse.ArgumentParser(
        description="Generate corrected PerpAdapt simulated pose trajectories."
    )
    parser.add_argument("--output", default="outputs/sim_clean.npz")
    parser.add_argument("--instances", type=int, default=1000)
    parser.add_argument("--samples-per-circle", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20261015)
    parser.add_argument(
        "--noise-mode",
        choices=NOISE_MODES,
        default="gaussian",
        help="none, Gaussian only, structured outliers only, or both",
    )
    parser.add_argument(
        "--noise-std",
        type=float,
        default=0.0,
        help="translation-noise standard deviation in metres (0.001 = 1 mm)",
    )
    parser.add_argument("--noise-seed", type=int, default=20261016)
    parser.add_argument(
        "--outlier-fraction",
        type=float,
        default=0.1,
        help="fraction of each trajectory placed in one contiguous outlier block",
    )
    parser.add_argument(
        "--outlier-offset",
        type=float,
        default=0.03,
        help="fixed offset magnitude for each structured block in metres",
    )
    parser.add_argument("--outlier-seed", type=int, default=20261017)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    dataset = generate_dataset(
        instances=args.instances,
        samples_per_circle=args.samples_per_circle,
        seed=args.seed,
        noise_std=args.noise_std,
        noise_seed=args.noise_seed,
        noise_mode=args.noise_mode,
        outlier_fraction=args.outlier_fraction,
        outlier_offset=args.outlier_offset,
        outlier_seed=args.outlier_seed,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, **dataset)

    print(f"saved: {output}")
    print(f"alpha_R: {dataset['alpha_R'].shape}")
    print(f"alpha_t: {dataset['alpha_t'].shape}")
    print(f"beta_R:  {dataset['beta_R'].shape}")
    print(f"beta_t:  {dataset['beta_t'].shape}")
    print(f"t:       {dataset['t'].shape}")
    print(
        "base angles (degrees): "
        f"alpha={ALPHA_BASE_DEG.tolist()}, beta={BETA_BASE_DEG.tolist()}"
    )
    print(f"noise mode: {dataset['noise_mode'].item()}")
    print(f"Gaussian translation noise std: {dataset['noise_std'].item():g} m")
    print(
        "structured outliers: "
        f"{dataset['outlier_count_per_circle'].item()} frames/trajectory, "
        f"offset={dataset['outlier_offset'].item():g} m"
    )


if __name__ == "__main__":
    main()

