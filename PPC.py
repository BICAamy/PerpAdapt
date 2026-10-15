"""Correct perpendicular-pivot calibration (PPC).

The original repository fitted a sphere to each trajectory and then used the
vector from the world origin to the fitted center as the axis direction.  This
implementation fits an actual 3-D circle, so every axis is represented by both
an anchor point and a direction and is independent of the chosen world origin.
"""

from __future__ import annotations

import time

import numpy as np

from geometry import (
    closest_points_between_lines,
    fit_circle_3d,
    inverse_transform_point,
    quaternion_to_rotation_matrix,
    regression_metrics,
)


def find_foot(point1, direction_vector1, point2, direction_vector2):
    """Compatibility wrapper returning the closest point on each input line."""
    return closest_points_between_lines(
        point1, direction_vector1, point2, direction_vector2
    )


def getT(rotation, position):
    """Compatibility helper constructing a world-from-sensor pose matrix."""
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.asarray(rotation, dtype=np.float64)
    transform[:3, 3] = np.asarray(position, dtype=np.float64)
    return transform


def step1(alpha_circles, beta_circles):
    """Fit the C-arm-axis and x-axis circles used by PPC."""
    c_alpha, n_alpha, _ = fit_circle_3d(np.asarray(alpha_circles)[0])
    c_beta, n_beta, _ = fit_circle_3d(np.asarray(beta_circles)[0])
    return c_alpha, n_alpha, c_beta, n_beta


def step2(c_alpha, n_alpha, c_beta, n_beta):
    """Return the endpoint of the common perpendicular on the C-arm axis."""
    c_arm_foot, _ = closest_points_between_lines(
        c_alpha, n_alpha, c_beta, n_beta
    )
    return c_arm_foot


def step3(transforms, pivot_world):
    """Average inverse-transformed pivots and return a homogeneous 4-vector."""
    transforms = np.asarray(transforms, dtype=np.float64).reshape(-1, 4, 4)
    estimates = []
    for transform in transforms:
        estimates.append(
            inverse_transform_point(
                transform[:3, :3], transform[:3, 3], pivot_world
            )
        )
    return np.append(np.mean(estimates, axis=0), 1.0)


def _filter_circle(sac, points, rotations=None):
    points = np.asarray(points, dtype=np.float64)
    if sac is None:
        return points, None if rotations is None else np.asarray(rotations)
    _, inliers = sac.fit(points)
    inliers = np.asarray(inliers)
    if inliers.dtype == bool:
        inliers = np.flatnonzero(inliers)
    if len(inliers) < 3:
        raise ValueError(f"sample consensus retained only {len(inliers)} circle points")
    filtered_rotations = None if rotations is None else np.asarray(rotations)[inliers]
    return points[inliers], filtered_rotations


def _first_circle(array, samples_per_circle=None):
    array = np.asarray(array)
    if array.ndim == 4:
        return array[:, 0]
    if array.ndim != 3:
        raise ValueError(f"expected (instances, samples, channels), got {array.shape}")
    if samples_per_circle is None:
        return array
    if array.shape[1] < samples_per_circle:
        raise ValueError("dataset contains fewer samples than samples_per_circle")
    return array[:, :samples_per_circle]


class PPC:
    """PPC using one beta-sweep circle and one alpha-sweep circle.

    ``sac`` may be ``None`` or any object exposing ``fit(points)`` and returning
    ``(model, inlier_indices)``.  Gaussian-noise-only experiments should use
    ``sac=None`` because there are no outliers to reject.
    """

    def __init__(self, sac=None):
        self.sac = sac

    def calibrate(self, dataset, *, verbose=True):
        samples_per_circle = None
        if "samples_per_circle" in dataset:
            samples_per_circle = int(np.asarray(dataset["samples_per_circle"]).item())

        alpha_quaternions = _first_circle(dataset["alpha_R"], samples_per_circle)
        alpha_positions = _first_circle(dataset["alpha_t"], samples_per_circle)
        beta_positions = _first_circle(dataset["beta_t"], samples_per_circle)
        truth = np.asarray(dataset["t"], dtype=np.float64)

        if not (
            len(alpha_quaternions)
            == len(alpha_positions)
            == len(beta_positions)
            == len(truth)
        ):
            raise ValueError("dataset arrays have inconsistent instance counts")

        predictions = []
        pivots = []
        x_axis_points = []
        x_axis_directions = []
        c_axis_points = []
        c_axis_directions = []
        sac_times = []
        calibration_times = []

        for index in range(len(truth)):
            sac_start = time.perf_counter()
            beta_points, _ = _filter_circle(self.sac, beta_positions[index])
            alpha_points, alpha_rotations = _filter_circle(
                self.sac, alpha_positions[index], alpha_quaternions[index]
            )
            sac_times.append(time.perf_counter() - sac_start if self.sac else 0.0)

            calibration_start = time.perf_counter()
            c_axis_point, c_axis_direction, _ = fit_circle_3d(alpha_points)
            x_axis_point, x_axis_direction, _ = fit_circle_3d(beta_points)

            # The pivot is the endpoint on the C-arm axis, not the point on the
            # x axis.  This also fixes the sign error in the old 2x2 system.
            pivot, _ = closest_points_between_lines(
                c_axis_point,
                c_axis_direction,
                x_axis_point,
                x_axis_direction,
            )

            local_estimates = []
            for quaternion, position in zip(alpha_rotations, alpha_points):
                rotation = quaternion_to_rotation_matrix(quaternion)
                local_estimates.append(
                    inverse_transform_point(rotation, position, pivot)
                )
            predictions.append(np.mean(local_estimates, axis=0))
            pivots.append(pivot)
            x_axis_points.append(x_axis_point)
            x_axis_directions.append(x_axis_direction)
            c_axis_points.append(c_axis_point)
            c_axis_directions.append(c_axis_direction)
            calibration_times.append(time.perf_counter() - calibration_start)

        prediction = np.asarray(predictions)
        pivot_prediction = np.asarray(pivots)
        metrics = regression_metrics(truth, prediction)
        result = {
            "method": "PPC",
            "t_pred": prediction,
            "cc_pred": pivot_prediction,
            "mae": metrics["mae"],
            "mse": metrics["mse"],
            "r2": metrics["r2"],
            "t_l2_mean": metrics["l2_mean"],
            "sac_time_mean": float(np.mean(sac_times)),
            "calibration_time_mean": float(np.mean(calibration_times)),
            "x_axis_points": np.asarray(x_axis_points),
            "x_axis_directions": np.asarray(x_axis_directions),
            "c_axis_points": np.asarray(c_axis_points),
            "c_axis_directions": np.asarray(c_axis_directions),
        }
        if "cc" in dataset:
            cc_truth = np.asarray(dataset["cc"], dtype=np.float64)
            result["cc_error_mean"] = float(
                np.mean(np.linalg.norm(pivot_prediction - cc_truth, axis=1))
            )
        elif "cc_all" in dataset:
            cc_truth = np.asarray(dataset["cc_all"], dtype=np.float64)[:, 0]
            result["cc_error_mean"] = float(
                np.mean(np.linalg.norm(pivot_prediction - cc_truth, axis=1))
            )

        if verbose:
            print("PPC")
            print(f"MAE: {result['mae']:.12g}")
            print(f"MSE: {result['mse']:.12g}")
            print(f"R2: {result['r2']:.12g}")
            if "cc_error_mean" in result:
                print(f"c_c,0 mean error: {result['cc_error_mean']:.12g}")
            print(f"calibration_time: {result['calibration_time_mean']:.6f} s/instance")
        return result
