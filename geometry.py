"""Shared geometry routines for PPC and Lee's perpendicular-circle baseline.

All poses use the convention ``p_world = R_world_sensor @ p_sensor + t_world``.
Quaternions are stored in SciPy/GitHub dataset order ``(x, y, z, w)``.
"""

from __future__ import annotations

import numpy as np


DEFAULT_MIN_TRIANGLE_QUALITY = 1e-2


class DegenerateCircleSample(ValueError):
    """Raised when three sampled points cannot define a stable circle."""


def unit(vector: np.ndarray, *, name: str = "vector") -> np.ndarray:
    """Return a normalized 3-vector and reject degenerate input."""
    vector = np.asarray(vector, dtype=np.float64)
    norm = np.linalg.norm(vector)
    if not np.isfinite(norm) or norm <= np.finfo(np.float64).eps:
        raise ValueError(f"{name} must have non-zero finite length")
    return vector / norm


def _as_points(points: np.ndarray, *, min_points: int = 3) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"expected points with shape (N, 3), got {points.shape}")
    if len(points) < min_points:
        raise ValueError(f"at least {min_points} points are required, got {len(points)}")
    if not np.all(np.isfinite(points)):
        raise ValueError("points contain NaN or infinity")
    return points


def triangle_quality(points: np.ndarray) -> float:
    """Return a rigid- and scale-invariant quality score for three points.

    The score is ``||e01 x e02|| / longest_edge^2``.  It is zero for
    collinear/repeated points and ``sqrt(3)/2`` for an equilateral triangle.
    Small values therefore identify numerically unstable minimal samples.
    """
    points = _as_points(points)
    if len(points) != 3:
        raise ValueError(f"triangle quality requires exactly 3 points, got {len(points)}")

    edge01 = points[1] - points[0]
    edge02 = points[2] - points[0]
    edge12 = points[2] - points[1]
    longest_edge_squared = max(
        float(np.dot(edge01, edge01)),
        float(np.dot(edge02, edge02)),
        float(np.dot(edge12, edge12)),
    )
    if longest_edge_squared <= np.finfo(np.float64).tiny:
        return 0.0
    return float(np.linalg.norm(np.cross(edge01, edge02)) / longest_edge_squared)


def circle_from_three_points(
    points: np.ndarray,
    *,
    min_triangle_quality: float = DEFAULT_MIN_TRIANGLE_QUALITY,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Construct the unique 3-D circle through a stable three-point sample.

    Unlike the old unconstrained variance minimization, the circumcenter is
    solved analytically inside the plane of the three points.
    """
    points = _as_points(points)
    if len(points) != 3:
        raise ValueError(f"minimal circle fitting requires exactly 3 points, got {len(points)}")
    if not 0.0 <= min_triangle_quality < 1.0:
        raise ValueError("min_triangle_quality must be in [0, 1)")

    quality = triangle_quality(points)
    if quality < min_triangle_quality:
        raise DegenerateCircleSample(
            "three-point circle sample is nearly collinear: "
            f"quality={quality:.6g} < {min_triangle_quality:.6g}"
        )

    point0, point1, point2 = points
    edge01 = point1 - point0
    edge02 = point2 - point0
    cross = np.cross(edge01, edge02)
    cross_squared = float(np.dot(cross, cross))
    if cross_squared <= np.finfo(np.float64).tiny:
        raise DegenerateCircleSample("three-point circle sample is collinear")

    offset = (
        np.dot(edge01, edge01) * np.cross(edge02, cross)
        + np.dot(edge02, edge02) * np.cross(cross, edge01)
    ) / (2.0 * cross_squared)
    center = point0 + offset
    normal = cross / np.sqrt(cross_squared)
    radius = float(np.linalg.norm(center - point0))
    if not np.all(np.isfinite(center)) or not np.isfinite(radius):
        raise DegenerateCircleSample("three-point circle construction is not finite")
    return center, normal, radius


def circle_residuals(model, points: np.ndarray) -> np.ndarray:
    """Euclidean point-to-circle residuals for a 3-D circle model."""
    center, normal, radius = model
    points = _as_points(points, min_points=1)
    center = np.asarray(center, dtype=np.float64)
    normal = unit(normal, name="circle normal")
    delta = points - center
    signed_plane_distance = delta @ normal
    projected_delta = delta - np.outer(signed_plane_distance, normal)
    radial_distance = np.linalg.norm(projected_delta, axis=1)
    return np.sqrt(
        signed_plane_distance * signed_plane_distance
        + (radial_distance - float(radius)) ** 2
    )


def draw_nondegenerate_triplet(
    points: np.ndarray,
    rng,
    *,
    min_triangle_quality: float = DEFAULT_MIN_TRIANGLE_QUALITY,
    probabilities: np.ndarray | None = None,
    max_attempts: int = 1000,
) -> tuple[np.ndarray, np.ndarray, int, float]:
    """Sample until a stable three-point circle hypothesis can be formed.

    Returns ``(indices, sampled_points, attempts_used, triangle_quality)``.
    """
    points = _as_points(points)
    if max_attempts < 1:
        raise ValueError("max_attempts must be positive")
    if not 0.0 <= min_triangle_quality < 1.0:
        raise ValueError("min_triangle_quality must be in [0, 1)")

    if probabilities is not None:
        probabilities = np.asarray(probabilities, dtype=np.float64)
        if probabilities.shape != (len(points),):
            raise ValueError("sampling probabilities must have shape (N,)")
        if not np.all(np.isfinite(probabilities)) or np.any(probabilities < 0.0):
            raise ValueError("sampling probabilities must be finite and non-negative")
        total = float(np.sum(probabilities))
        if total <= 0.0 or np.count_nonzero(probabilities) < 3:
            probabilities = None
        else:
            probabilities = probabilities / total

    for attempt in range(1, max_attempts + 1):
        indices = rng.choice(len(points), 3, replace=False, p=probabilities)
        sample = points[indices]
        quality = triangle_quality(sample)
        if quality < min_triangle_quality:
            continue
        try:
            # Keep exact/repeated-point rejection active even when callers set
            # min_triangle_quality=0 to disable only the near-collinear margin.
            circle_from_three_points(
                sample, min_triangle_quality=min_triangle_quality
            )
        except DegenerateCircleSample:
            continue
        return np.asarray(indices), sample, attempt, quality

    raise DegenerateCircleSample(
        f"failed to draw a stable three-point sample in {max_attempts} attempts "
        f"(minimum quality {min_triangle_quality:.6g})"
    )


def _plane_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Build a stable orthonormal basis of the plane perpendicular to normal."""
    normal = unit(normal, name="plane normal")
    helper = np.eye(3, dtype=np.float64)[np.argmin(np.abs(normal))]
    e1 = unit(np.cross(normal, helper), name="plane basis")
    e2 = np.cross(normal, e1)
    return e1, e2


def _fit_circle_2d(xy: np.ndarray) -> tuple[np.ndarray, float]:
    """Linear least-squares circle fit for coordinates in one plane."""
    design = np.column_stack((2.0 * xy[:, 0], 2.0 * xy[:, 1], np.ones(len(xy))))
    rhs = np.einsum("ij,ij->i", xy, xy)
    solution, _, rank, _ = np.linalg.lstsq(design, rhs, rcond=None)
    if rank < 3:
        raise ValueError("circle fit is degenerate; points are collinear or repeated")
    cx, cy, constant = solution
    radius_squared = cx * cx + cy * cy + constant
    tolerance = 100.0 * np.finfo(np.float64).eps * max(1.0, np.max(rhs))
    if radius_squared < -tolerance:
        raise ValueError("circle fit produced a negative squared radius")
    return np.array([cx, cy], dtype=np.float64), float(np.sqrt(max(0.0, radius_squared)))


def fit_circle_3d(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Fit a genuine 3-D circle: SVD plane fit followed by a 2-D circle fit.

    This is intentionally different from fitting a sphere.  The smallest right
    singular vector is the circle-plane normal; the circle center is solved only
    inside that plane.
    """
    points = _as_points(points)
    centroid = np.mean(points, axis=0)
    centered = points - centroid
    _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    scale = max(float(singular_values[0]), 1.0)
    if singular_values[1] <= 100.0 * np.finfo(np.float64).eps * scale:
        raise ValueError("circle fit is degenerate; points do not span a plane")

    e1, e2, normal = vt[0], vt[1], unit(vt[2], name="circle normal")
    xy = np.column_stack((centered @ e1, centered @ e2))
    center_2d, radius = _fit_circle_2d(xy)
    center = centroid + center_2d[0] * e1 + center_2d[1] * e2
    return center, normal, radius


def fit_circle_with_normal(
    points: np.ndarray, normal: np.ndarray
) -> tuple[np.ndarray, float]:
    """Fit a circle to 3-D points in a plane with a known normal."""
    points = _as_points(points)
    normal = unit(normal, name="known circle normal")
    e1, e2 = _plane_basis(normal)
    centroid = np.mean(points, axis=0)
    centered = points - centroid
    xy = np.column_stack((centered @ e1, centered @ e2))
    center_2d, radius = _fit_circle_2d(xy)
    center = centroid + center_2d[0] * e1 + center_2d[1] * e2
    return center, radius


def align_and_average(normals: list[np.ndarray] | np.ndarray) -> np.ndarray:
    """Resolve the sign ambiguity of fitted normals before averaging them."""
    normalized = [unit(n, name="circle normal") for n in normals]
    if not normalized:
        raise ValueError("at least one normal is required")
    reference = normalized[0]
    aligned = [n if np.dot(n, reference) >= 0.0 else -n for n in normalized]
    return unit(np.mean(aligned, axis=0), name="mean circle normal")


def closest_points_between_lines(
    point_a: np.ndarray,
    direction_a: np.ndarray,
    point_b: np.ndarray,
    direction_b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the endpoints of the common perpendicular of two 3-D lines."""
    point_a = np.asarray(point_a, dtype=np.float64)
    point_b = np.asarray(point_b, dtype=np.float64)
    direction_a = unit(direction_a, name="first line direction")
    direction_b = unit(direction_b, name="second line direction")

    dot_ab = float(np.dot(direction_a, direction_b))
    matrix = np.array([[1.0, -dot_ab], [dot_ab, -1.0]], dtype=np.float64)
    rhs_delta = point_b - point_a
    rhs = np.array(
        [np.dot(rhs_delta, direction_a), np.dot(rhs_delta, direction_b)],
        dtype=np.float64,
    )
    parameters, _, rank, _ = np.linalg.lstsq(matrix, rhs, rcond=None)
    if rank < 2:
        raise ValueError("rotation axes are parallel; the common perpendicular is not unique")
    parameter_a, parameter_b = parameters
    return (
        point_a + parameter_a * direction_a,
        point_b + parameter_b * direction_b,
    )


def project_point_to_plane(
    point: np.ndarray, plane_point: np.ndarray, plane_normal: np.ndarray
) -> np.ndarray:
    normal = unit(plane_normal, name="plane normal")
    point = np.asarray(point, dtype=np.float64)
    plane_point = np.asarray(plane_point, dtype=np.float64)
    return point - np.dot(point - plane_point, normal) * normal


def closest_point_on_line(
    point: np.ndarray, line_point: np.ndarray, line_direction: np.ndarray
) -> np.ndarray:
    direction = unit(line_direction, name="line direction")
    point = np.asarray(point, dtype=np.float64)
    line_point = np.asarray(line_point, dtype=np.float64)
    return line_point + np.dot(point - line_point, direction) * direction


def quaternion_to_rotation_matrix(quaternion: np.ndarray) -> np.ndarray:
    """Convert one ``(x, y, z, w)`` quaternion to a rotation matrix."""
    quaternion = np.asarray(quaternion, dtype=np.float64)
    if quaternion.shape != (4,):
        raise ValueError(f"expected quaternion with shape (4,), got {quaternion.shape}")
    quaternion = unit(quaternion, name="quaternion")
    x, y, z, w = quaternion
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def inverse_transform_point(
    rotation_world_sensor: np.ndarray,
    translation_world_sensor: np.ndarray,
    point_world: np.ndarray,
) -> np.ndarray:
    """Transform a world-coordinate point into the sensor coordinate system."""
    rotation = np.asarray(rotation_world_sensor, dtype=np.float64)
    translation = np.asarray(translation_world_sensor, dtype=np.float64)
    point = np.asarray(point_world, dtype=np.float64)
    return rotation.T @ (point - translation)


def infer_samples_per_circle(dataset: dict, total_alpha_samples: int) -> int:
    """Read v2 metadata, otherwise infer the legacy three-circle layout."""
    if "samples_per_circle" in dataset:
        value = int(np.asarray(dataset["samples_per_circle"]).item())
        if value < 3:
            raise ValueError("samples_per_circle must be at least 3")
        return value
    if total_alpha_samples % 3 != 0:
        raise ValueError(
            "legacy PC input must contain three equally sized alpha circles; "
            "add samples_per_circle metadata for other layouts"
        )
    return total_alpha_samples // 3


def regression_metrics(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    """Compute component-wise MAE/MSE and sklearn-compatible uniform-average R²."""
    truth = np.asarray(truth, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    if truth.shape != prediction.shape:
        raise ValueError(f"truth shape {truth.shape} != prediction shape {prediction.shape}")
    residual = prediction - truth
    mae = float(np.mean(np.abs(residual)))
    mse = float(np.mean(residual * residual))
    l2_mean = float(np.mean(np.linalg.norm(residual, axis=1)))

    numerator = np.sum(residual * residual, axis=0)
    centered = truth - np.mean(truth, axis=0)
    denominator = np.sum(centered * centered, axis=0)
    if np.any(denominator == 0.0):
        r2 = float("nan")
    else:
        r2 = float(np.mean(1.0 - numerator / denominator))
    return {"mae": mae, "mse": mse, "r2": r2, "l2_mean": l2_mean}
