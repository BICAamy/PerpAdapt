import unittest

import numpy as np

from geometry import (
    DegenerateCircleSample,
    circle_from_three_points,
    draw_nondegenerate_triplet,
    triangle_quality,
)
from evaluate_corrected import build_parser as build_evaluation_parser
from evaluate_corrected import evaluate
from PerpAdapt import build_parser
from RANSAC import RANSAC
from simulation import generate_dataset


class _ChoiceSequence:
    """Small deterministic RNG stub for exercising the retry path."""

    def __init__(self, choices):
        self.choices = [np.asarray(choice, dtype=np.int64) for choice in choices]

    def choice(self, _count, _size, replace=False, p=None):
        self.last_replace = replace
        self.last_probabilities = p
        return self.choices.pop(0)


class _AllInlierSAC:
    def __init__(self):
        self.fit_calls = 0

    def fit(self, points):
        self.fit_calls += 1
        return None, np.arange(len(points), dtype=np.int64)


class SampleConsensusTests(unittest.TestCase):
    def setUp(self):
        self.points = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 1e-10, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [-1.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )

    def test_nearly_collinear_triplet_is_rejected(self):
        sample = self.points[:3]
        self.assertLess(triangle_quality(sample), 1e-2)
        with self.assertRaises(DegenerateCircleSample):
            circle_from_three_points(sample, min_triangle_quality=1e-2)

    def test_stable_triplet_gives_the_in_plane_circle(self):
        center, normal, radius = circle_from_three_points(self.points[3:])
        np.testing.assert_allclose(center, np.zeros(3), atol=1e-14)
        self.assertAlmostEqual(radius, 1.0, places=14)
        self.assertAlmostEqual(abs(normal[2]), 1.0, places=14)

    def test_triangle_quality_is_rigid_and_scale_invariant(self):
        sample = self.points[3:]
        rotation = np.array(
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
        )
        transformed = 7.5 * (sample @ rotation.T) + np.array([5.0, -3.0, 2.0])
        self.assertAlmostEqual(
            triangle_quality(sample), triangle_quality(transformed), places=14
        )

    def test_sampler_retries_after_degenerate_draw(self):
        rng = _ChoiceSequence(([0, 1, 2], [3, 4, 5]))
        indices, sample, attempts, quality = draw_nondegenerate_triplet(
            self.points,
            rng,
            min_triangle_quality=1e-2,
            max_attempts=2,
        )
        np.testing.assert_array_equal(indices, [3, 4, 5])
        np.testing.assert_array_equal(sample, self.points[3:])
        self.assertEqual(attempts, 2)
        self.assertGreaterEqual(quality, 1e-2)

    def test_ransac_counts_rejected_draw_and_keeps_valid_hypothesis(self):
        ransac = RANSAC(
            num_iterations=1,
            sample_size=3,
            distance_threshold=1e-8,
            min_triangle_quality=1e-2,
            max_sampling_attempts=2,
            random_state=0,
        )
        ransac.rng = _ChoiceSequence(([0, 1, 2], [3, 4, 5]))
        model, inliers = ransac.fit(self.points)
        np.testing.assert_allclose(model[0], np.zeros(3), atol=1e-14)
        self.assertGreaterEqual(len(inliers), 3)
        self.assertEqual(
            ransac.last_sampling_stats,
            {"attempts": 2, "degenerate_samples": 1, "valid_hypotheses": 1},
        )

    def test_cli_accepts_requested_sac_names_case_insensitively(self):
        parser = build_parser()
        expected = {
            "None": "none",
            "RANSAC": "ransac",
            "MemorySAC": "memorysac",
        }
        for supplied, normalized in expected.items():
            with self.subTest(supplied=supplied):
                self.assertEqual(
                    parser.parse_args(["--sac", supplied]).sac, normalized
                )

    def test_evaluation_cli_accepts_requested_sac_names(self):
        parser = build_evaluation_parser()
        expected = {
            "None": "none",
            "RANSAC": "ransac",
            "MemorySAC": "memorysac",
        }
        for supplied, normalized in expected.items():
            with self.subTest(supplied=supplied):
                args = parser.parse_args(
                    ["--dataset", "data.npz", "--sac", supplied]
                )
                self.assertEqual(args.sac, normalized)

    def test_evaluation_applies_selected_sac_to_all_three_methods(self):
        dataset = generate_dataset(
            instances=1,
            samples_per_circle=12,
            seed=20261015,
            noise_std=0.0,
        )
        samplers = []

        def factory():
            sampler = _AllInlierSAC()
            samplers.append(sampler)
            return sampler

        results = evaluate(dataset, sac_factory=factory)
        self.assertEqual(
            [result["method"] for result in results],
            ["PPC", "Lee(4)", "Lee(6)"],
        )
        self.assertEqual(len(samplers), 3)
        self.assertEqual([sampler.fit_calls for sampler in samplers], [2, 4, 6])


if __name__ == "__main__":
    unittest.main()
