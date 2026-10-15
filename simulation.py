"""Generate reproducible PerpAdapt simulation datasets.

The fixed base angles are specified in degrees and converted exactly once to
radians.  Optional noise is added only to the measured world translations;
quaternions and clean ground-truth labels remain unchanged.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


ALPHA_BASE_DEG = np.array([0.0, 90.0, -90.0], dtype=np.float64)
BETA_BASE_DEG = np.array([0.0, 45.0, -45.0], dtype=np.float64)


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


def generate_dataset(
    instances=1000,
    samples_per_circle=100,
    seed=20261015,
    noise_std=0.0,
    noise_seed=20261016,
):
    """Generate arrays compatible with the original ``dataset.npz`` layout.

    The coordinate unit is metres.  For example, ``noise_std=0.001`` means
    independent 1 mm standard-deviation Gaussian noise in every translation
    component of every pose.
    """
    if instances < 1:
        raise ValueError("instances must be positive")
    if samples_per_circle < 3:
        raise ValueError("samples_per_circle must be at least 3")
    if noise_std < 0.0:
        raise ValueError("noise_std cannot be negative")

    data_rng = np.random.default_rng(seed)
    noise_rng = np.random.default_rng(noise_seed)
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
        if noise_std > 0.0:
            # Keep this per-instance alpha-then-beta order so a fixed noise seed
            # reproduces the documented experiments exactly.
            alpha_points = alpha_points + noise_rng.normal(
                0.0, noise_std, size=alpha_points.shape
            )
            beta_points = beta_points + noise_rng.normal(
                0.0, noise_std, size=beta_points.shape
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
        "samples_per_circle": np.array(samples_per_circle, dtype=np.int64),
        "alpha_base_deg": ALPHA_BASE_DEG.copy(),
        "beta_base_deg": BETA_BASE_DEG.copy(),
        "seed": np.array(seed, dtype=np.int64),
        "noise_seed": np.array(noise_seed, dtype=np.int64),
        "noise_std": np.array(noise_std, dtype=np.float64),
        "position_unit": np.array("m"),
        "format_version": np.array("perpadapt-sim-v2"),
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
        "--noise-std",
        type=float,
        default=0.0,
        help="translation-noise standard deviation in metres (0.001 = 1 mm)",
    )
    parser.add_argument("--noise-seed", type=int, default=20261016)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    dataset = generate_dataset(
        instances=args.instances,
        samples_per_circle=args.samples_per_circle,
        seed=args.seed,
        noise_std=args.noise_std,
        noise_seed=args.noise_seed,
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
    print(f"translation noise std: {args.noise_std:g} m")


if __name__ == "__main__":
    main()
