"""Tests for perception/orientation.py: every branch of the A.3 precedence ladder, using stubs.

Run: python perception/test_orientation.py -v
"""
import sys
import unittest
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import orientation as ori  # noqa: E402
import render_compare as rc  # noqa: E402
from test_render_compare import stepped_building, stepped_mask_seen_from_front  # noqa: E402

UP, DOWN = (0, 1, 0), (0, -1, 0)


@dataclass(frozen=True)
class PhotoEvidenceStub:
    """Same shape as ML_ADDENDUM F.1's PhotoEvidence (contracts.py is frozen and not importable here)."""

    mask: np.ndarray
    mask_area_frac: float
    occlusion_flag: bool
    exif_heading_deg: float | None
    exif_pitch_deg: float | None
    silhouette_scores: dict
    silhouette_margin: float


def photo(heading=None):
    return PhotoEvidenceStub(np.zeros((4, 4), bool), 0.5, False, heading, None, {}, 0.99)


def facade_heading(candidate):
    """Stub for geo/: azimuth 0/90/180/270 -> the photographed facade faces 30/120/210/300."""
    return (candidate.azimuth_deg + 30.0) % 360.0


def comparison(scores=(0.9, 0.5, 0.4, 0.3), margin=None, label=None, extra=()):
    """A real render_compare.Comparison over up=+Y at azimuths 0/90/180/270, plus any `extra` candidates."""
    cands = [rc.Candidate(UP, az, s) for az, s in zip((0.0, 90.0, 180.0, 270.0), scores)] + list(extra)
    ranked = sorted(cands, key=lambda c: -c.score)
    m = ranked[0].score - ranked[1].score if margin is None else margin
    return rc.Comparison(ranked, ranked[0], ranked[1], m, 0, label=label, needs_review=m < rc.REVIEW_MARGIN)


def az(choice):
    return choice.candidate.azimuth_deg


class ExifBranchTests(unittest.TestCase):
    def test_exif_decides_and_needs_no_review(self):
        # camera points at 30 -> photographed facade faces 210 -> the candidate at azimuth 180.
        r = ori.choose_orientation(photo(30.0), comparison(), facade_heading)
        self.assertEqual((az(r), r.disambiguated_by, r.needs_review), (180.0, "exif_heading", False))
        self.assertIn("exif heading 30.0", r.reason)

    def test_exif_beats_a_confident_silhouette_that_disagrees(self):
        c = comparison(scores=(0.95, 0.4, 0.3, 0.2))  # margin 0.55, silhouette says azimuth 0
        self.assertEqual(c.best.azimuth_deg, 0.0)
        r = ori.choose_orientation(photo(30.0), c, facade_heading)
        self.assertEqual((az(r), r.disambiguated_by), (180.0, "exif_heading"))

    def test_exif_beats_a_low_margin_and_a_road_normal(self):
        r = ori.choose_orientation(photo(30.0), comparison(margin=0.0), facade_heading, road_normal_deg=300.0)
        self.assertEqual((az(r), r.disambiguated_by), (180.0, "exif_heading"))

    def test_heading_zero_is_present_and_wraps_correctly(self):
        # 0.0 is falsy but valid: faces 180 -> nearest is 210 (30 off) -> azimuth 180.
        r = ori.choose_orientation(photo(0.0), comparison(), facade_heading)
        self.assertEqual((az(r), r.disambiguated_by), (180.0, "exif_heading"))
        # 359 -> faces 179 -> 210 is 31 off, 120 is 59 off.
        self.assertEqual(az(ori.choose_orientation(photo(359.0), comparison(), facade_heading)), 180.0)
        # 300 -> faces 120 -> azimuth 90 exactly.
        r = ori.choose_orientation(photo(300.0), comparison(), facade_heading)
        self.assertEqual(az(r), 90.0)
        self.assertIn("0.0 deg off", r.reason)

    def test_exif_only_chooses_among_the_silhouette_bests_up_axis(self):
        extra = [rc.Candidate(DOWN, 180.0, 0.9)]  # same facade bearing as (UP, 180), better score than it
        c = comparison(scores=(0.95, 0.5, 0.2, 0.3), extra=extra)
        self.assertEqual(c.best.up_axis, UP)
        r = ori.choose_orientation(photo(30.0), c, facade_heading)
        self.assertEqual((r.candidate.up_axis, az(r), r.candidate.score), (UP, 180.0, 0.2))

    def test_exact_tie_between_two_facades_goes_to_the_better_silhouette(self):
        # facing 75 is 45 from both 30 (az 0, score .6) and 120 (az 90, score .7).
        r = ori.choose_orientation(photo(255.0), comparison(scores=(0.6, 0.7, 0.1, 0.1)), facade_heading)
        self.assertEqual(az(r), 90.0)
        r = ori.choose_orientation(photo(255.0), comparison(scores=(0.7, 0.6, 0.1, 0.1)), facade_heading)
        self.assertEqual(az(r), 0.0)


class DisagreementTests(unittest.TestCase):
    """exif_silhouette_disagree: EXIF decided AND margin >= 0.08 AND the silhouette's best differs. EXIF still wins."""

    def test_confident_silhouette_that_disagrees_is_recorded_but_exif_still_wins(self):
        c = comparison(scores=(0.95, 0.4, 0.3, 0.2))  # silhouette: azimuth 0, margin 0.55
        r = ori.choose_orientation(photo(30.0), c, facade_heading)  # EXIF: azimuth 180
        self.assertEqual((az(r), r.disambiguated_by, r.needs_review), (180.0, "exif_heading", False))
        self.assertTrue(r.exif_silhouette_disagree)
        self.assertAlmostEqual(r.silhouette_margin, 0.55)
        self.assertIn("DISAGREES", r.reason)

    def test_agreement_is_not_a_disagreement(self):
        c = comparison(scores=(0.4, 0.3, 0.95, 0.2))  # silhouette: azimuth 180, margin 0.55
        r = ori.choose_orientation(photo(30.0), c, facade_heading)  # EXIF: azimuth 180
        self.assertEqual(az(r), 180.0)
        self.assertFalse(r.exif_silhouette_disagree)
        self.assertNotIn("DISAGREES", r.reason)

    def test_an_unconfident_silhouette_cannot_disagree(self):
        c = comparison(scores=(0.6, 0.55, 0.5, 0.2))  # best az 0, margin 0.05
        r = ori.choose_orientation(photo(30.0), c, facade_heading)  # EXIF: azimuth 180
        self.assertEqual(az(r), 180.0)
        self.assertFalse(r.exif_silhouette_disagree)
        self.assertAlmostEqual(r.silhouette_margin, 0.05)

    def test_threshold_is_0_08_inclusive(self):
        self.assertTrue(ori.choose_orientation(photo(30.0), comparison(margin=0.08), facade_heading).exif_silhouette_disagree)
        self.assertFalse(ori.choose_orientation(photo(30.0), comparison(margin=0.0799), facade_heading).exif_silhouette_disagree)

    def test_only_the_exif_branch_can_disagree_and_the_margin_is_always_recorded(self):
        confident, weak = comparison(scores=(0.9, 0.5, 0.4, 0.3)), comparison(margin=0.03)
        cases = [
            (ori.choose_orientation(photo(None), confident), 0.4),
            (ori.choose_orientation(photo(None), weak, facade_heading, road_normal_deg=120.0), 0.03),
            (ori.choose_orientation(photo(None), weak), 0.03),
            (ori.choose_orientation(photo(30.0), confident, facade_heading), 0.4),
        ]
        for r, margin in cases:
            self.assertAlmostEqual(r.silhouette_margin, margin)
        self.assertEqual([r.exif_silhouette_disagree for r, _ in cases], [False, False, False, True])


class ImplausibleExifTests(unittest.TestCase):
    def test_implausible_or_absent_headings_are_ignored_and_the_ladder_falls_through(self):
        confident = comparison(scores=(0.9, 0.5, 0.4, 0.3))  # margin 0.4 -> silhouette
        for bad in (-5.0, 360.0, 400.0, float("nan"), float("inf"), "north", True):
            r = ori.choose_orientation(photo(bad), confident, facade_heading)
            self.assertEqual((az(r), r.disambiguated_by), (0.0, "silhouette"), bad)
            self.assertIn("implausible", r.reason, bad)
        r = ori.choose_orientation(photo(None), confident, facade_heading)
        self.assertEqual(r.disambiguated_by, "silhouette")
        self.assertNotIn("implausible", r.reason)

    def test_plausible_heading_boundaries(self):
        self.assertEqual(ori.plausible_heading(0), 0.0)
        self.assertEqual(ori.plausible_heading("142.5"), 142.5)
        self.assertEqual(ori.plausible_heading(359.99), 359.99)
        self.assertIsNone(ori.plausible_heading(360))
        self.assertIsNone(ori.plausible_heading(None))


class SilhouetteBranchTests(unittest.TestCase):
    def test_confident_silhouette_decides_without_a_facade_mapping(self):
        r = ori.choose_orientation(photo(None), comparison(scores=(0.9, 0.5, 0.4, 0.3)))
        self.assertEqual((az(r), r.disambiguated_by, r.needs_review), (0.0, "silhouette", False))

    def test_threshold_is_0_08_inclusive(self):
        self.assertEqual(ori.SILHOUETTE_DECISIVE_MARGIN, 0.08)
        at = ori.choose_orientation(photo(), comparison(margin=0.08), facade_heading, road_normal_deg=120.0)
        self.assertEqual(at.disambiguated_by, "silhouette")
        below = ori.choose_orientation(photo(), comparison(margin=0.0799), facade_heading, road_normal_deg=120.0)
        self.assertEqual(below.disambiguated_by, "road_normal")

    def test_a_road_normal_is_ignored_when_the_silhouette_is_confident(self):
        r = ori.choose_orientation(photo(), comparison(), facade_heading, road_normal_deg=300.0)
        self.assertEqual((az(r), r.disambiguated_by), (0.0, "silhouette"))


class RoadNormalBranchTests(unittest.TestCase):
    def test_road_normal_picks_the_nearest_facade_and_always_flags_review(self):
        c = comparison(scores=(0.99, 0.98, 0.5, 0.5), margin=0.03)  # near-perfect IoU, still a guess
        r = ori.choose_orientation(photo(), c, facade_heading, road_normal_deg=100.0)  # nearest: 120 (az 90)
        self.assertEqual((az(r), r.disambiguated_by, r.needs_review), (90.0, "road_normal", True))
        self.assertIn("guessing", r.reason)

    def test_bearing_wraps_and_normalises(self):
        c = comparison(margin=0.0)
        self.assertEqual(az(ori.choose_orientation(photo(), c, facade_heading, road_normal_deg=359.0)), 0.0)  # 359~30: 31
        self.assertEqual(az(ori.choose_orientation(photo(), c, facade_heading, road_normal_deg=-60.0)), 270.0)  # 300
        self.assertEqual(az(ori.choose_orientation(photo(), c, facade_heading, road_normal_deg=750.0)), 0.0)  # 30

    def test_road_normal_is_used_when_exif_is_implausible(self):
        r = ori.choose_orientation(photo(400.0), comparison(margin=0.03), facade_heading, road_normal_deg=210.0)
        self.assertEqual((az(r), r.disambiguated_by, r.needs_review), (180.0, "road_normal", True))
        self.assertIn("implausible", r.reason)

    def test_tie_goes_to_the_better_silhouette_and_stays_in_the_best_up_axis(self):
        extra = [rc.Candidate(DOWN, 90.0, 0.6)]
        c = comparison(scores=(0.6, 0.55, 0.1, 0.1), margin=0.0, extra=extra)
        r = ori.choose_orientation(photo(), c, facade_heading, road_normal_deg=75.0)  # 45 from az 0 and az 90
        self.assertEqual((r.candidate.up_axis, az(r)), (UP, 0.0))


class NoEvidenceTests(unittest.TestCase):
    def test_no_exif_low_margin_and_no_road_normal_is_flagged_unresolved(self):
        c = comparison(margin=0.03)
        r = ori.choose_orientation(photo(), c, facade_heading)
        self.assertEqual((r.candidate, r.disambiguated_by, r.needs_review), (c.best, "unresolved", True))
        self.assertIn("no road normal", r.reason)

    def test_symmetric_label_is_recorded_when_nothing_else_can_decide(self):
        c = comparison(margin=0.0, label=rc.ARBITRARY_SYMMETRIC)
        r = ori.choose_orientation(photo(), c)
        self.assertEqual((r.disambiguated_by, r.needs_review), ("arbitrary_symmetric", True))
        self.assertEqual(ori.ARBITRARY_SYMMETRIC, rc.ARBITRARY_SYMMETRIC)

    def test_a_symmetric_building_with_a_road_normal_uses_the_road(self):
        r = ori.choose_orientation(photo(), comparison(margin=0.0, label=rc.ARBITRARY_SYMMETRIC), facade_heading,
                                   road_normal_deg=120.0)
        self.assertEqual((az(r), r.disambiguated_by, r.needs_review), (90.0, "road_normal", True))


class FailLoudlyTests(unittest.TestCase):
    def test_evidence_without_a_facade_mapping_raises_instead_of_being_skipped(self):
        with self.assertRaisesRegex(ValueError, "EXIF heading.*facade_heading"):
            ori.choose_orientation(photo(30.0), comparison())
        with self.assertRaisesRegex(ValueError, "road normal.*facade_heading"):
            ori.choose_orientation(photo(), comparison(margin=0.0), road_normal_deg=100.0)

    def test_nonfinite_inputs_from_geo_raise(self):
        with self.assertRaisesRegex(ValueError, "road_normal_deg"):
            ori.choose_orientation(photo(), comparison(margin=0.0), facade_heading, road_normal_deg=float("nan"))
        with self.assertRaisesRegex(ValueError, "facade_heading returned"):
            ori.choose_orientation(photo(30.0), comparison(), lambda c: float("nan"))

    def test_photo_without_the_exif_field_is_a_shape_error(self):
        with self.assertRaises(AttributeError):
            ori.choose_orientation(object(), comparison(), facade_heading)


class HelperTests(unittest.TestCase):
    def test_bearing_from_enu(self):
        for (e, n), expected in {(0, 1): 0, (1, 0): 90, (0, -1): 180, (-1, 0): 270, (1, 1): 45}.items():
            self.assertAlmostEqual(ori.bearing_from_enu(e, n), expected)

    def test_angular_difference_wraps(self):
        self.assertEqual(ori.angular_diff_deg(359, 1), 2)
        self.assertEqual(ori.angular_diff_deg(0, 180), 180)
        self.assertEqual(ori.angular_diff_deg(90, 90), 0)


class WithARealComparisonTests(unittest.TestCase):
    def test_works_on_a_genuine_render_compare_result(self):
        real = rc.render_compare(stepped_building(), stepped_mask_seen_from_front())
        self.assertGreater(real.margin, ori.SILHOUETTE_DECISIVE_MARGIN)
        r = ori.choose_orientation(photo(None), real)
        self.assertEqual((r.candidate, r.disambiguated_by, r.needs_review), (real.best, "silhouette", False))
        # Same real comparison, but a camera heading that says the opposite facade: EXIF wins.
        # Here the mesh's azimuth-0 facade faces 0 and azimuth k faces 90*k, so heading 0 -> facade 180.
        r = ori.choose_orientation(photo(0.0), real, lambda c: c.azimuth_deg)
        self.assertEqual((az(r), r.disambiguated_by), (180.0, "exif_heading"))


if __name__ == "__main__":
    unittest.main()
