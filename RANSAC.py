"""RANSAC circle inlier selection with degenerate-sample rejection."""

from __future__ import annotations

import numpy as np

from geometry import (
    DEFAULT_MIN_TRIANGLE_QUALITY,
    DegenerateCircleSample,
    circle_from_three_points,
    circle_residuals,
    draw_nondegenerate_triplet,
)


def variance_of_distances(center, points):
    distances = np.linalg.norm(np.asarray(points) - np.asarray(center), axis=1)
    return float(np.var(distances))


def circumcenter(points):
    """Compatibility helper returning the stable in-plane circumcenter."""
    center, _, _ = circle_from_three_points(points)
    return center


def model_func(vertices, min_triangle_quality=DEFAULT_MIN_TRIANGLE_QUALITY):
    return circle_from_three_points(
        vertices, min_triangle_quality=min_triangle_quality
    )


def distance_func(model, point):
    return float(circle_residuals(model, np.asarray(point).reshape(1, 3))[0])


class RANSAC:
    def __init__(
        self,
        num_iterations,
        sample_size,
        distance_threshold,
        min_triangle_quality=DEFAULT_MIN_TRIANGLE_QUALITY,
        max_sampling_attempts=1000,
        random_state=None,
    ):
        if num_iterations < 1:
            raise ValueError("num_iterations must be positive")
        if sample_size != 3:
            raise ValueError("3-D circle hypotheses require sample_size=3")
        if distance_threshold < 0.0:
            raise ValueError("distance_threshold cannot be negative")
        if not 0.0 <= min_triangle_quality < 1.0:
            raise ValueError("min_triangle_quality must be in [0, 1)")
        if max_sampling_attempts < num_iterations:
            raise ValueError("max_sampling_attempts must be >= num_iterations")

        self.num_iterations = int(num_iterations)
        self.sample_size = int(sample_size)
        self.distance_threshold = float(distance_threshold)
        self.min_triangle_quality = float(min_triangle_quality)
        self.max_sampling_attempts = int(max_sampling_attempts)
        self.rng = np.random.default_rng(random_state)
        self.last_sampling_stats = None

    def fit(self, data, model_func=model_func, distance_func=distance_func):
        data = np.asarray(data, dtype=np.float64)
        if data.ndim != 2 or data.shape[1] != 3 or len(data) < 3:
            raise ValueError(f"expected at least three points with shape (N, 3), got {data.shape}")
        if not np.all(np.isfinite(data)):
            raise ValueError("input points contain NaN or infinity")

        best_model = None
        best_inliers = np.empty(0, dtype=np.int64)
        best_residual_sum = float("inf")
        total_attempts = 0
        degenerate_samples = 0
        valid_hypotheses = 0

        while valid_hypotheses < self.num_iterations:
            remaining_attempts = self.max_sampling_attempts - total_attempts
            if remaining_attempts <= 0:
                break
            try:
                _, sample, attempts_used, _ = draw_nondegenerate_triplet(
                    data,
                    self.rng,
                    min_triangle_quality=self.min_triangle_quality,
                    max_attempts=remaining_attempts,
                )
            except DegenerateCircleSample:
                degenerate_samples += remaining_attempts
                total_attempts = self.max_sampling_attempts
                break

            total_attempts += attempts_used
            degenerate_samples += attempts_used - 1
            if model_func is globals()["model_func"]:
                candidate_model = model_func(
                    sample, min_triangle_quality=self.min_triangle_quality
                )
            else:
                candidate_model = model_func(sample)
            valid_hypotheses += 1

            if distance_func is globals()["distance_func"]:
                distances = circle_residuals(candidate_model, data)
            else:
                distances = np.asarray(
                    [distance_func(candidate_model, point) for point in data],
                    dtype=np.float64,
                )
            inliers = np.flatnonzero(distances < self.distance_threshold)
            residual_sum = float(np.sum(distances[inliers]))
            if (
                best_model is None
                or len(inliers) > len(best_inliers)
                or (
                    len(inliers) == len(best_inliers)
                    and residual_sum < best_residual_sum
                )
            ):
                best_model = candidate_model
                best_inliers = inliers
                best_residual_sum = residual_sum

        self.last_sampling_stats = {
            "attempts": total_attempts,
            "degenerate_samples": degenerate_samples,
            "valid_hypotheses": valid_hypotheses,
        }
        if valid_hypotheses < self.num_iterations or best_model is None:
            raise RuntimeError(
                "RANSAC could not draw enough non-degenerate three-point samples: "
                f"valid={valid_hypotheses}/{self.num_iterations}, "
                f"attempts={total_attempts}, minimum_quality={self.min_triangle_quality:g}"
            )
        return best_model, best_inliers
