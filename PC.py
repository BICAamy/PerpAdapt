"""Corrected implementation of Lee et al.'s perpendicular-circle baseline.

PC(4) uses three C-arm-axis circles plus one x-axis circle.  PC(6) uses the
same three C-arm-axis circles plus three x-axis circles.
"""

from __future__ import annotations

import time

import numpy as np

from geometry import (
    align_and_average,
    closest_point_on_line,
    fit_circle_3d,
    fit_circle_with_normal,
    infer_samples_per_circle,
    inverse_transform_point,
    project_point_to_plane,
    quaternion_to_rotation_matrix,
    regression_metrics,
    unit,
)


def project_to_plane(point, plane_point, plane_normal):
    """Compatibility wrapper with the corrected dot-product projection."""
    return project_point_to_plane(point, plane_point, plane_normal)


def point_to_line_distance(point, line_point, line_direction):
    closest = closest_point_on_line(point, line_point, line_direction)
    return float(np.linalg.norm(np.asarray(point) - closest))


def getT(rotation, position):
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.asarray(rotation, dtype=np.float64)
    transform[:3, 3] = np.asarray(position, dtype=np.float64)
    return transform


def computecc(nc, cc, C, R=None):
    """Return the midpoint of line/sphere intersections in homogeneous form.

    The midpoint is simply the closest point from sphere center ``C`` to the
    line ``cc + lambda * nc``; the radius is irrelevant to that midpoint.
    """
    point = closest_point_on_line(C, cc, nc)
    return np.append(point, 1.0)


def step1(beta_circles):
    x_centers = []
    x_normals = []
    for circle in beta_circles:
        center, normal, _ = fit_circle_3d(circle)
        x_centers.append(center)
        x_normals.append(normal)
    return np.mean(x_centers, axis=0), align_and_average(x_normals)


def step2(cx, n, alpha_circles):
    c_centers = []
    c_normals = []
    for circle in alpha_circles:
        center, normal, _ = fit_circle_3d(circle)
        c_centers.append(center)
        c_normals.append(normal)
    c_centers = np.asarray(c_centers)

    # Lee's construction fits the trajectory of the three minor-circle centers.
    fitted_center, _ = fit_circle_with_normal(c_centers, n)
    projected_x_center = project_point_to_plane(cx, fitted_center, n)
    torus_center = 0.5 * (fitted_center + projected_x_center)
    return c_centers, np.asarray(c_normals), torus_center


def step3(cc_list, nc_list, c, n=None):
    return float(
        np.mean(
            [
                point_to_line_distance(c, circle_center, circle_normal)
                for circle_center, circle_normal in zip(cc_list, nc_list)
            ]
        )
    )


def step4(transforms, nc_list, cc_list, c, r=None):
    per_circle = []
    for circle_transforms, normal, center in zip(transforms, nc_list, cc_list):
        pivot = closest_point_on_line(c, center, normal)
        estimates = []
        for transform in circle_transforms:
            transform = np.asarray(transform, dtype=np.float64)
            estimates.append(
                inverse_transform_point(
                    transform[:3, :3], transform[:3, 3], pivot
                )
            )
        per_circle.append(np.mean(estimates, axis=0))
    average = np.mean(per_circle, axis=0)
    return [np.append(value, 1.0) for value in per_circle] + [np.append(average, 1.0)]


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


def _reshape_circles(array, circle_count, samples_per_circle):
    array = np.asarray(array)
    if array.ndim == 4:
        if array.shape[1] < circle_count:
            raise ValueError(f"expected at least {circle_count} circles, got {array.shape[1]}")
        return array[:, :circle_count, :samples_per_circle]
    if array.ndim != 3:
        raise ValueError(f"expected flattened or grouped circle data, got {array.shape}")
    required = circle_count * samples_per_circle
    if array.shape[1] < required:
        raise ValueError(f"expected at least {required} samples, got {array.shape[1]}")
    return array[:, :required].reshape(
        array.shape[0], circle_count, samples_per_circle, array.shape[2]
    )


class PC:
    """Lee PC baseline with either four or six total circles."""

    def __init__(self, sac=None, pc_circle_num=4):
        if pc_circle_num not in (4, 6):
            raise ValueError("pc_circle_num must be 4 or 6")
        self.sac = sac
        self.pc_circle_num = pc_circle_num

    def calibrate(self, dataset, *, verbose=True):
        alpha_positions_raw = np.asarray(dataset["alpha_t"])
        if alpha_positions_raw.ndim == 4:
            samples_per_circle = alpha_positions_raw.shape[2]
        else:
            samples_per_circle = infer_samples_per_circle(
                dataset, alpha_positions_raw.shape[1]
            )

        x_circle_count = self.pc_circle_num - 3
        alpha_positions = _reshape_circles(
            alpha_positions_raw, 3, samples_per_circle
        )
        alpha_quaternions = _reshape_circles(
            dataset["alpha_R"], 3, samples_per_circle
        )
        beta_positions = _reshape_circles(
            dataset["beta_t"], x_circle_count, samples_per_circle
        )
        truth = np.asarray(dataset["t"], dtype=np.float64)

        if not (
            len(alpha_positions)
            == len(alpha_quaternions)
            == len(beta_positions)
            == len(truth)
        ):
            raise ValueError("dataset arrays have inconsistent instance counts")

        predictions = []
        pivots_all = []
        torus_centers = []
        torus_normals = []
        major_radii = []
        sac_times = []
        calibration_times = []

        for index in range(len(truth)):
            sac_start = time.perf_counter()
            filtered_beta = [
                _filter_circle(self.sac, circle)[0]
                for circle in beta_positions[index]
            ]
            filtered_alpha = []
            filtered_quaternions = []
            for circle, rotations in zip(
                alpha_positions[index], alpha_quaternions[index]
            ):
                points, quaternions = _filter_circle(
                    self.sac, circle, rotations
                )
                filtered_alpha.append(points)
                filtered_quaternions.append(quaternions)
            sac_times.append(time.perf_counter() - sac_start if self.sac else 0.0)

            calibration_start = time.perf_counter()
            x_centers = []
            x_normals = []
            for circle in filtered_beta:
                center, normal, _ = fit_circle_3d(circle)
                x_centers.append(center)
                x_normals.append(normal)
            torus_normal = align_and_average(x_normals)
            x_center = np.mean(x_centers, axis=0)

            c_centers = []
            c_normals = []
            for circle in filtered_alpha:
                center, normal, _ = fit_circle_3d(circle)
                c_centers.append(center)
                c_normals.append(normal)
            c_centers = np.asarray(c_centers)
            c_normals = np.asarray(c_normals)

            fitted_center, _ = fit_circle_with_normal(c_centers, torus_normal)
            projected_x_center = project_point_to_plane(
                x_center, fitted_center, torus_normal
            )
            torus_center = 0.5 * (fitted_center + projected_x_center)

            pivots = np.asarray(
                [
                    closest_point_on_line(torus_center, center, normal)
                    for center, normal in zip(c_centers, c_normals)
                ]
            )
            radii = np.linalg.norm(pivots - torus_center, axis=1)

            local_estimates = []
            for pivot, quaternions, positions in zip(
                pivots, filtered_quaternions, filtered_alpha
            ):
                for quaternion, position in zip(quaternions, positions):
                    rotation = quaternion_to_rotation_matrix(quaternion)
                    local_estimates.append(
                        inverse_transform_point(rotation, position, pivot)
                    )

            predictions.append(np.mean(local_estimates, axis=0))
            pivots_all.append(pivots)
            torus_centers.append(torus_center)
            torus_normals.append(torus_normal)
            major_radii.append(float(np.mean(radii)))
            calibration_times.append(time.perf_counter() - calibration_start)

        prediction = np.asarray(predictions)
        pivot_prediction_all = np.asarray(pivots_all)
        metrics = regression_metrics(truth, prediction)
        result = {
            "method": f"Lee({self.pc_circle_num})",
            "t_pred": prediction,
            "cc_pred": pivot_prediction_all[:, 0],
            "cc_all_pred": pivot_prediction_all,
            "mae": metrics["mae"],
            "mse": metrics["mse"],
            "r2": metrics["r2"],
            "t_l2_mean": metrics["l2_mean"],
            "sac_time_mean": float(np.mean(sac_times)),
            "calibration_time_mean": float(np.mean(calibration_times)),
            "torus_centers": np.asarray(torus_centers),
            "torus_normals": np.asarray(torus_normals),
            "major_radii": np.asarray(major_radii),
        }
        if "cc" in dataset:
            cc_truth = np.asarray(dataset["cc"], dtype=np.float64)
            result["cc_error_mean"] = float(
                np.mean(np.linalg.norm(result["cc_pred"] - cc_truth, axis=1))
            )
        elif "cc_all" in dataset:
            cc_truth_all = np.asarray(dataset["cc_all"], dtype=np.float64)
            result["cc_error_mean"] = float(
                np.mean(
                    np.linalg.norm(pivot_prediction_all[:, 0] - cc_truth_all[:, 0], axis=1)
                )
            )
            result["cc_all_error_mean"] = float(
                np.mean(np.linalg.norm(pivot_prediction_all - cc_truth_all, axis=2))
            )

        if verbose:
            print(result["method"])
            print(f"MAE: {result['mae']:.12g}")
            print(f"MSE: {result['mse']:.12g}")
            print(f"R2: {result['r2']:.12g}")
            if "cc_error_mean" in result:
                print(f"c_c,0 mean error: {result['cc_error_mean']:.12g}")
            print(f"calibration_time: {result['calibration_time_mean']:.6f} s/instance")
        return result
