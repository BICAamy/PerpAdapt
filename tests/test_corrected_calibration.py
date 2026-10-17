import unittest

import numpy as np

from geometry import closest_points_between_lines
from PC import PC
from PPC import PPC
from simulation import (
    ALPHA_BASE_DEG,
    BETA_BASE_DEG,
    build_parser as build_simulation_parser,
    generate_dataset,
)


class CorrectedCalibrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.clean = generate_dataset(
            instances=8,
            samples_per_circle=30,
            seed=20261015,
            noise_std=0.0,
        )

    def test_clean_data_reaches_floating_point_precision(self):
        results = [
            PPC().calibrate(self.clean, verbose=False),
            PC(pc_circle_num=4).calibrate(self.clean, verbose=False),
            PC(pc_circle_num=6).calibrate(self.clean, verbose=False),
        ]
        for result in results:
            self.assertLess(result["cc_error_mean"], 1e-12, result["method"])
            self.assertLess(result["mse"], 1e-24, result["method"])

    def test_fixed_angles_are_degrees_converted_once(self):
        np.testing.assert_allclose(
            self.clean["alpha_base_deg"], ALPHA_BASE_DEG, atol=0.0, rtol=0.0
        )
        np.testing.assert_allclose(
            self.clean["beta_base_deg"], BETA_BASE_DEG, atol=0.0, rtol=0.0
        )
        self.assertEqual(int(self.clean["samples_per_circle"]), 30)

    def test_translation_noise_does_not_change_rotations_or_truth(self):
        noisy = generate_dataset(
            instances=2,
            samples_per_circle=10,
            seed=123,
            noise_std=0.001,
            noise_seed=456,
        )
        clean = generate_dataset(
            instances=2,
            samples_per_circle=10,
            seed=123,
            noise_std=0.0,
            noise_seed=456,
        )
        np.testing.assert_array_equal(noisy["alpha_R"], clean["alpha_R"])
        np.testing.assert_array_equal(noisy["beta_R"], clean["beta_R"])
        np.testing.assert_array_equal(noisy["t"], clean["t"])
        np.testing.assert_array_equal(noisy["cc_all"], clean["cc_all"])
        self.assertFalse(np.array_equal(noisy["alpha_t"], clean["alpha_t"]))
        self.assertFalse(np.array_equal(noisy["beta_t"], clean["beta_t"]))

    def test_explicit_gaussian_mode_preserves_legacy_generation(self):
        legacy = generate_dataset(
            instances=2,
            samples_per_circle=10,
            seed=123,
            noise_std=0.001,
            noise_seed=456,
        )
        explicit = generate_dataset(
            instances=2,
            samples_per_circle=10,
            seed=123,
            noise_mode="gaussian",
            noise_std=0.001,
            noise_seed=456,
        )
        np.testing.assert_array_equal(legacy["alpha_t"], explicit["alpha_t"])
        np.testing.assert_array_equal(legacy["beta_t"], explicit["beta_t"])

    def test_structured_mode_adds_contiguous_constant_offset_blocks(self):
        samples = 20
        clean = generate_dataset(
            instances=2,
            samples_per_circle=samples,
            seed=123,
            noise_mode="none",
        )
        structured = generate_dataset(
            instances=2,
            samples_per_circle=samples,
            seed=123,
            noise_mode="structured",
            outlier_fraction=0.2,
            outlier_offset=0.03,
            outlier_seed=789,
        )

        np.testing.assert_array_equal(structured["alpha_R"], clean["alpha_R"])
        np.testing.assert_array_equal(structured["beta_R"], clean["beta_R"])
        np.testing.assert_array_equal(structured["t"], clean["t"])
        np.testing.assert_array_equal(structured["cc_all"], clean["cc_all"])
        self.assertEqual(structured["noise_mode"].item(), "structured")
        self.assertEqual(int(structured["outlier_count_per_circle"]), 4)

        for prefix in ("alpha", "beta"):
            mask = structured[f"{prefix}_outlier_mask"].reshape(2, 3, samples)
            offsets = structured[f"{prefix}_outlier_offsets"]
            starts = structured[f"{prefix}_outlier_starts"]
            positions = structured[f"{prefix}_t"].reshape(2, 3, samples, 3)
            clean_positions = clean[f"{prefix}_t"].reshape(2, 3, samples, 3)
            delta = positions - clean_positions
            np.testing.assert_array_equal(mask.sum(axis=2), np.full((2, 3), 4))

            for instance in range(2):
                for circle in range(3):
                    expected_indices = (
                        starts[instance, circle] + np.arange(4)
                    ) % samples
                    np.testing.assert_array_equal(
                        np.flatnonzero(mask[instance, circle]),
                        np.sort(expected_indices),
                    )
                    np.testing.assert_allclose(
                        delta[instance, circle][mask[instance, circle]],
                        np.repeat(offsets[instance, circle][None, :], 4, axis=0),
                        atol=1e-15,
                    )
                    np.testing.assert_array_equal(
                        delta[instance, circle][~mask[instance, circle]],
                        np.zeros((samples - 4, 3)),
                    )
                    self.assertAlmostEqual(
                        np.linalg.norm(offsets[instance, circle]), 0.03, places=14
                    )

    def test_both_mode_adds_outliers_on_top_of_same_gaussian_samples(self):
        common = {
            "instances": 1,
            "samples_per_circle": 20,
            "seed": 123,
            "noise_std": 0.001,
            "noise_seed": 456,
        }
        gaussian = generate_dataset(noise_mode="gaussian", **common)
        both = generate_dataset(
            noise_mode="both",
            outlier_fraction=0.25,
            outlier_offset=0.02,
            outlier_seed=789,
            **common,
        )
        for prefix in ("alpha", "beta"):
            delta = both[f"{prefix}_t"] - gaussian[f"{prefix}_t"]
            mask = both[f"{prefix}_outlier_mask"]
            np.testing.assert_array_equal(delta[~mask], np.zeros_like(delta[~mask]))
            self.assertEqual(int(mask.sum()), 3 * 5)
        self.assertEqual(both["noise_mode"].item(), "both")
        self.assertEqual(float(both["noise_std"]), 0.001)

    def test_noise_mode_command_line_choices(self):
        parser = build_simulation_parser()
        for mode in ("none", "gaussian", "structured", "both"):
            with self.subTest(mode=mode):
                self.assertEqual(
                    parser.parse_args(["--noise-mode", mode]).noise_mode,
                    mode,
                )

    def test_common_perpendicular_uses_offset_axis_lines(self):
        point_c, point_x = closest_points_between_lines(
            np.array([2.0, 3.0, 4.0]),
            np.array([0.0, 1.0, 0.0]),
            np.array([5.0, -1.0, 1.0]),
            np.array([1.0, 0.0, 0.0]),
        )
        np.testing.assert_allclose(point_c, [2.0, -1.0, 4.0], atol=1e-14)
        np.testing.assert_allclose(point_x, [2.0, -1.0, 1.0], atol=1e-14)


if __name__ == "__main__":
    unittest.main()

