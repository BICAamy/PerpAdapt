"""MemorySAC inlier selection with stable three-point circle hypotheses."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torchdiffeq import odeint

from geometry import (
    DEFAULT_MIN_TRIANGLE_QUALITY,
    DegenerateCircleSample,
    circle_from_three_points,
    circle_residuals,
    draw_nondegenerate_triplet,
)


class Config:
    device = torch.device("cpu")
    t0 = 0
    tbar = 2
    tpoints = 10
    n_samples = 100
    inputsize = 3
    outputsize = 2
    batchsize = 32
    lr = 0.01
    num_workers = 2
    epoch = 100


config = Config()


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


class WeightedAveragePooling(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weights = torch.nn.Parameter(torch.ones(dim))

    def forward(self, inputs):
        normalized_weights = F.softmax(self.weights, dim=0).view(-1, 1, 1, 1)
        return torch.sum(normalized_weights * inputs, dim=0)


class neuralODE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gamma = None

    def fresh(self, gamma):
        self.gamma = gamma

    def forward(self, t, p):
        return -p + torch.pow(torch.sin(p + self.gamma), 2)


class nmODE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.neuralODE = neuralODE()
        self.t = torch.tensor(
            np.linspace(config.t0, config.tbar, config.tpoints),
            device=config.device,
        )
        self.w1 = torch.nn.Linear(config.inputsize, 256)
        self.w2 = torch.nn.Linear(256, 1024)
        # Kept for checkpoint compatibility; forward uses torch.max directly.
        self.pool = torch.nn.MaxPool1d(kernel_size=config.n_samples)
        self.w3 = torch.nn.Linear(1024 + 256, 128)
        self.out = torch.nn.Linear(128, config.outputsize)
        self.wpool = WeightedAveragePooling(config.tpoints)

    def forward(self, x):
        feature = F.relu(self.w1(x))
        self.neuralODE.fresh(feature)
        y0 = torch.zeros(
            x.shape[0], x.shape[1], 256, device=x.device, dtype=x.dtype
        )
        yt = odeint(
            self.neuralODE,
            y0,
            self.t.to(device=x.device, dtype=x.dtype),
            method="rk4",
            options={"step_size": 0.1},
        )
        yt = self.wpool(yt)

        global_feature = F.relu(self.w2(feature))
        global_feature, _ = torch.max(global_feature, dim=1, keepdim=True)
        global_feature = global_feature.repeat(1, feature.shape[1], 1)
        combined = torch.cat((global_feature, yt), dim=2)
        combined = F.relu(self.w3(combined))
        return self.out(combined)


class MemorySAC:
    def __init__(
        self,
        num_iterations,
        sample_size,
        distance_threshold,
        update_rate,
        pth,
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
        if update_rate < 0.0:
            raise ValueError("update_rate cannot be negative")
        if not 0.0 <= min_triangle_quality < 1.0:
            raise ValueError("min_triangle_quality must be in [0, 1)")
        if max_sampling_attempts < num_iterations:
            raise ValueError("max_sampling_attempts must be >= num_iterations")

        self.num_iterations = int(num_iterations)
        self.sample_size = int(sample_size)
        self.distance_threshold = float(distance_threshold)
        self.update_rate = float(update_rate)
        self.min_triangle_quality = float(min_triangle_quality)
        self.max_sampling_attempts = int(max_sampling_attempts)
        self.rng = np.random.default_rng(random_state)
        self.last_sampling_stats = None

        self.net = nmODE()
        try:
            state = torch.load(
                pth, map_location=config.device, weights_only=True
            )
        except TypeError:  # PyTorch versions without weights_only.
            state = torch.load(pth, map_location=config.device)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        self.net.load_state_dict(state)
        self.net.eval()

    def _initial_scores(self, data):
        tensor = torch.as_tensor(data, dtype=torch.float32, device=config.device)
        with torch.no_grad():
            logits = self.net(tensor.unsqueeze(0))
            probabilities = F.softmax(logits, dim=-1)[0, :, 1]
        scores = probabilities.detach().cpu().numpy().astype(np.float64)
        return np.maximum(np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0), 0.0)

    def fit(self, data, model_func=model_func, distance_func=distance_func):
        data = np.asarray(data, dtype=np.float64)
        if data.ndim != 2 or data.shape[1] != 3 or len(data) < 3:
            raise ValueError(f"expected at least three points with shape (N, 3), got {data.shape}")
        if not np.all(np.isfinite(data)):
            raise ValueError("input points contain NaN or infinity")

        memory_scores = self._initial_scores(data)
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
                    probabilities=memory_scores,
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

            memory_scores = np.maximum(
                memory_scores - self.update_rate * distances, 0.0
            )
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
                "MemorySAC could not draw enough non-degenerate three-point samples: "
                f"valid={valid_hypotheses}/{self.num_iterations}, "
                f"attempts={total_attempts}, minimum_quality={self.min_triangle_quality:g}"
            )
        return best_model, best_inliers
