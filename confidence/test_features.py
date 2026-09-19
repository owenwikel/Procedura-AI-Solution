"""Tests for confidence/features.py.

Run: python confidence/test_features.py -v
"""
import math
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import features as feat  # noqa: E402


def fit(**kw):
    base = dict(iou=0.8, hausdorff_m=2.0, area_ratio=1.0, rotation_margin_footprint=0.2, anisotropy_log_ratio=0.0, max_neighbor_overlap=0.01)
    return NS(**{**base, **kw})


PHOTO, FP = NS(silhouette_margin=0.15), NS(rectilinearity=0.93)


def vec(f=None, **kw):
    kw = {"geocode_rooftop": True, "height_source_authoritative": False, **kw}
    return feat.feature_dict(f or fit(), PHOTO, FP, **kw)


class TableTests(unittest.TestCase):
    def test_b3_features_and_expected_signs(self):
        self.assertEqual(len(feat.FEATURE_NAMES), 10)
        self.assertEqual(set(feat.EXPECTED_SIGN), set(feat.FEATURE_NAMES))
        self.assertEqual({n for n, s in feat.EXPECTED_SIGN.items() if s < 0},
                         {"hausdorff_m", "abs_area_ratio_log", "abs_anisotropy_log_ratio", "max_neighbor_overlap"})

    def test_reduced_set_is_the_b4_four(self):
        self.assertEqual(set(feat.REDUCED_FEATURES), {"footprint_iou", "hausdorff_m", "rectilinearity", "rotation_margin_footprint"})


class ValueTests(unittest.TestCase):
    def test_each_feature_comes_from_the_right_place(self):
        d = vec(fit(iou=0.71, hausdorff_m=3.5, rotation_margin_footprint=0.33, max_neighbor_overlap=0.04))
        self.assertEqual(d["footprint_iou"], 0.71)
        self.assertEqual(d["hausdorff_m"], 3.5)
        self.assertEqual(d["rotation_margin_footprint"], 0.33)
        self.assertEqual(d["rotation_margin_silhouette"], 0.15)  # photo.silhouette_margin
        self.assertEqual(d["rectilinearity"], 0.93)  # fp.rectilinearity
        self.assertEqual(d["max_neighbor_overlap"], 0.04)
        self.assertEqual((d["geocode_rooftop"], d["height_source_authoritative"]), (1.0, 0.0))

    def test_area_ratio_log_is_symmetric_about_one(self):
        self.assertAlmostEqual(vec(fit(area_ratio=2.0))["abs_area_ratio_log"], math.log(2))
        self.assertAlmostEqual(vec(fit(area_ratio=0.5))["abs_area_ratio_log"], math.log(2))
        self.assertEqual(vec(fit(area_ratio=1.0))["abs_area_ratio_log"], 0.0)

    def test_anisotropy_is_a_magnitude(self):
        self.assertAlmostEqual(vec(fit(anisotropy_log_ratio=-0.3))["abs_anisotropy_log_ratio"], 0.3)
        self.assertAlmostEqual(vec(fit(anisotropy_log_ratio=0.3))["abs_anisotropy_log_ratio"], 0.3)

    def test_binaries_accept_bools_and_01_only(self):
        for value, expected in ((True, 1.0), (False, 0.0), (np.bool_(True), 1.0), (1, 1.0), (0, 0.0)):
            self.assertEqual(vec(geocode_rooftop=value)["geocode_rooftop"], expected)


class VectorTests(unittest.TestCase):
    def test_vector_follows_the_requested_names_in_order(self):
        kw = dict(geocode_rooftop=True, height_source_authoritative=False)
        full = feat.feature_vector(fit(), PHOTO, FP, **kw)
        self.assertEqual(full.shape, (10,))
        reduced = feat.feature_vector(fit(), PHOTO, FP, names=feat.REDUCED_FEATURES, **kw)
        d = vec()
        np.testing.assert_allclose(reduced, [d[n] for n in feat.REDUCED_FEATURES])
        np.testing.assert_allclose(full, [d[n] for n in feat.FEATURE_NAMES])

    def test_matrix_over_fake_rows(self):
        from fixtures.fake_fits import generate

        rows = generate(n_buildings=4, n_configs=2, n_corruptions=3, seed=1)
        X = feat.feature_matrix(rows)
        self.assertEqual(X.shape, (11, 10))
        self.assertTrue(np.isfinite(X).all())
        self.assertEqual(feat.feature_matrix(rows, feat.REDUCED_FEATURES).shape, (11, 4))


class FailLoudlyTests(unittest.TestCase):
    def test_nonpositive_or_nonfinite_area_ratio(self):
        for bad in (0.0, -1.0, float("nan"), float("inf")):
            with self.assertRaises(ValueError, msg=bad):
                vec(fit(area_ratio=bad))

    def test_nonfinite_metrics(self):
        for kw in (dict(iou=float("nan")), dict(hausdorff_m=float("inf")), dict(anisotropy_log_ratio=float("nan")),
                   dict(max_neighbor_overlap=float("nan")), dict(rotation_margin_footprint=float("inf"))):
            with self.assertRaisesRegex(ValueError, "not finite"):
                vec(fit(**kw))
        with self.assertRaisesRegex(ValueError, "not finite"):
            feat.feature_dict(fit(), NS(silhouette_margin=float("nan")), FP, geocode_rooftop=True, height_source_authoritative=True)
        with self.assertRaisesRegex(ValueError, "not finite"):
            feat.feature_dict(fit(), PHOTO, NS(rectilinearity=float("nan")), geocode_rooftop=True, height_source_authoritative=True)

    def test_binaries_must_be_boolean(self):
        for bad in ("yes", 2, None, 0.5, -1):
            with self.assertRaisesRegex(ValueError, "must be a bool"):
                vec(geocode_rooftop=bad)
            with self.assertRaisesRegex(ValueError, "must be a bool"):
                vec(height_source_authoritative=bad)

    def test_binaries_are_required_keywords(self):
        with self.assertRaises(TypeError):
            feat.feature_dict(fit(), PHOTO, FP)

    def test_missing_attribute_and_unknown_feature(self):
        with self.assertRaises(AttributeError):
            feat.feature_dict(NS(iou=0.8), PHOTO, FP, geocode_rooftop=True, height_source_authoritative=True)
        with self.assertRaisesRegex(KeyError, "unknown feature"):
            feat.feature_vector(fit(), PHOTO, FP, geocode_rooftop=True, height_source_authoritative=True, names=("nope",))


if __name__ == "__main__":
    unittest.main()
