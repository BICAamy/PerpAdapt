import unittest

import numpy as np

from geometry import closest_points_between_lines
from PC import PC
from PPC import PPC
from simulation import ALPHA_BASE_DEG, BETA_BASE_DEG, generate_dataset


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
