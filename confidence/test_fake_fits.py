"""Tests for fixtures/fake_fits.py (the synthetic data behind confidence/train.py).

Run: python confidence/test_fake_fits.py -v
"""
import dataclasses
import math
import sys
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import features as feat  # noqa: E402
from fixtures import fake_fits as ff  # noqa: E402

# Field names exactly as ML_ADDENDUM F.1 lists them.
F1 = {
    "FitResult": ["theta", "scale_x", "scale_y", "tx", "ty", "z_offset", "iou", "hausdorff_m", "area_ratio",
                  "rotation_margin_footprint", "anisotropy_log_ratio", "max_neighbor_overlap", "disambiguated_by"],
    "PhotoEvidence": ["mask", "mask_area_frac", "occlusion_flag", "exif_heading_deg", "exif_pitch_deg", "silhouette_scores", "silhouette_margin"],
    "Footprint": ["pts_enu", "rectilinearity", "ombb", "has_holes"],
}


class ContractShapeTests(unittest.TestCase):
    def test_classes_match_ml_addendum_f1(self):
        for name, fields in F1.items():
            self.assertEqual([f.name for f in dataclasses.fields(getattr(ff, name))], fields, name)


class GenerateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = ff.generate(seed=0)

    def test_default_layout_follows_b2(self):
        self.assertEqual(len(self.rows), 20 * 4 + 40)
        bench = [r for r in self.rows if r.config_index is not None]
        corrupt = [r for r in self.rows if r.config_index is None]
        self.assertEqual((len(bench), len(corrupt)), (80, 40))
        self.assertEqual({r.config_index for r in bench}, {0, 1, 2, 3})
        self.assertEqual({r.kind for r in corrupt}, set(ff.CORRUPTIONS))
        self.assertTrue(all(r.label == 0 and not r.label_flipped for r in corrupt))

    def test_is_deterministic_per_seed(self):
        a, b, c = ff.generate(seed=0), ff.generate(seed=0), ff.generate(seed=1)
        np.testing.assert_array_equal(feat.feature_matrix(a), feat.feature_matrix(b))
        self.assertFalse(np.array_equal(feat.feature_matrix(a), feat.feature_matrix(c)))
        self.assertEqual([r.label for r in a], [r.label for r in b])

    def test_classes_are_present_but_not_cleanly_separable(self):
        labels = np.array([r.label for r in self.rows])
        self.assertTrue(0.3 < labels.mean() < 0.6)
        self.assertTrue(any(r.label_flipped for r in self.rows))  # annotator disagreement is simulated
        self.assertTrue(any(r.kind == "borderline" for r in self.rows))

    def test_better_solver_configs_are_accepted_more_often(self):
        rate = lambda c: np.mean([r.label for r in self.rows if r.config_index == c])
        self.assertLess(rate(0), rate(3))

    def test_metrics_carry_the_expected_signal(self):
        X, y = feat.feature_matrix(self.rows), np.array([r.label for r in self.rows])
        col = {n: i for i, n in enumerate(feat.FEATURE_NAMES)}
        self.assertGreater(X[y == 1, col["footprint_iou"]].mean() - X[y == 0, col["footprint_iou"]].mean(), 0.15)
        self.assertLess(X[y == 1, col["hausdorff_m"]].mean(), X[y == 0, col["hausdorff_m"]].mean())
        self.assertLess(X[y == 1, col["abs_area_ratio_log"]].mean(), X[y == 0, col["abs_area_ratio_log"]].mean())

    def test_values_are_internally_coherent(self):
        for r in self.rows:
            f, p, fp = r.fit, r.photo, r.footprint
            self.assertAlmostEqual(f.area_ratio, f.scale_x * f.scale_y, places=9)
            self.assertAlmostEqual(f.anisotropy_log_ratio, math.log(f.scale_x / f.scale_y), places=9)
            top = sorted(p.silhouette_scores.values(), reverse=True)
            self.assertAlmostEqual(p.silhouette_margin, top[0] - top[1], places=9)
            self.assertEqual(set(p.silhouette_scores), {(0, k) for k in range(4)})  # (up_axis_idx, azimuth_k)
            self.assertAlmostEqual(p.mask_area_frac, float(p.mask.mean()))
            (_, _), (_, _), (a, b) = fp.ombb
            self.assertGreaterEqual(a, b)
            self.assertTrue(0 <= f.iou <= 1 and 0 <= f.rotation_margin_footprint <= 1 and 0 <= f.max_neighbor_overlap <= 1)
            self.assertIn(f.disambiguated_by, {"exif_heading", "silhouette", "road_normal"})
            self.assertEqual(p.exif_heading_deg is None or 0 <= p.exif_heading_deg < 360, True)

    def test_near_square_buildings_have_aspect_under_1_1_and_tiny_margins(self):
        square = [r for r in self.rows if r.footprint.ombb[2][0] / r.footprint.ombb[2][1] < 1.1]
        other = [r for r in self.rows if r.footprint.ombb[2][0] / r.footprint.ombb[2][1] >= 1.1]
        self.assertTrue(square and other)
        # Near-square footprints: the four OMBB candidates ~tie. (A mirrored-mesh corruption draws its margin
        # from U(0, 0.1) whatever the building, so it is excluded.)
        own = [r for r in square if r.kind != "corrupt_mirrored"]
        self.assertLessEqual(max(r.fit.rotation_margin_footprint for r in own), 0.06)
        self.assertGreater(np.mean([r.fit.rotation_margin_footprint for r in other]), 0.1)

    def test_features_are_finite_for_every_row(self):
        self.assertTrue(np.isfinite(feat.feature_matrix(self.rows)).all())

    def test_bad_label_noise_is_rejected(self):
        with self.assertRaises(ValueError):
            ff.generate(label_noise=0.6)


if __name__ == "__main__":
    unittest.main()
