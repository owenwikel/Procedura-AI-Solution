"""Tests for perception/segment.py using synthetic masks (no models, no photos).

Run: python perception/test_segment.py -v
Grounding DINO and SAM 2 are stubbed, so this exercises clean(), the A.4 retry,
the exit-1 paths and the cache without a GPU.
"""
import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import segment as s  # noqa: E402

W, H = 400, 300
DINO_BOX = [20.0, 20.0, 380.0, 280.0]
CENTRE_BOX = [40.0, 30.0, 360.0, 270.0]  # central 80% of a 400x300 frame
DETECTIONS = [{"box": DINO_BOX, "score": 0.9, "label": "building"}]


def rect(y0, y1, x0, x1):
    m = np.zeros((H, W), bool)
    m[y0:y1, x0:x1] = True
    return m


GOOD = rect(60, 240, 80, 320)
# A cross reaching all four edges; ~50% of the frame, so only the edge rule fires.
FOUR_EDGES = rect(100, 200, 0, W) | rect(0, H, 150, 250)
# 96% of the frame but inset from every edge, so only the coverage rule fires.
COVER_95 = rect(3, 297, 4, 396)
# Two 180 px tall blobs 40 px apart (71% of the larger), inside the DINO box, touching no edge.
TWO_BLOB = rect(60, 240, 60, 200) | rect(60, 240, 240, 340)


def components(mask):
    return cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)[0] - 1


def fake_sam(*masks):
    """Stand-in for sam_session(): serves queued masks and records each (box, point) prompt."""
    queue, calls = list(masks), []

    @contextlib.contextmanager
    def session(image):
        def segment(box, point=None):
            calls.append((box, point))
            return queue.pop(0)

        yield segment

    return session, calls


class CleanTests(unittest.TestCase):
    def test_merges_qualifying_component_and_drops_the_rest(self):
        box = [50, 50, 350, 250]
        m = rect(60, 240, 60, 200)  # largest: 25,200 px
        m |= rect(60, 240, 240, 340)  # 71% of largest, inside box -> merged
        m |= rect(52, 57, 62, 100)  # 190 px, <1% of largest, inside the box -> dropped (too small)
        big_outside = rect(260, 300, 0, 300)  # 12,000 px = 48% of largest, but box ends at y=250 -> dropped
        out, stats = s.clean(m | big_outside, box)
        self.assertEqual(stats["components_total"], 4)
        self.assertEqual(stats["components_merged"], 1)
        self.assertTrue(out[150, 100] and out[150, 300])
        self.assertFalse(out[54, 80] or out[280, 100])
        expected = (190 + 12000) / (25200 + 18000 + 190 + 12000)
        self.assertAlmostEqual(stats["discarded_area_frac"], expected, places=4)

    def test_merge_threshold_is_10_percent_of_the_largest(self):
        self.assertEqual(s.MERGE_MIN_FRACTION, 0.10)
        largest = rect(60, 240, 60, 200)  # 25,200 px
        above = largest | rect(60, 240, 240, 255)  # 2,700 px = 10.7% -> merged
        below = largest | rect(60, 240, 240, 252)  # 2,160 px = 8.6% -> discarded
        self.assertEqual(s.clean(above, DINO_BOX)[1]["components_merged"], 1)
        self.assertEqual(s.clean(below, DINO_BOX)[1]["components_merged"], 0)

    def test_component_straddling_the_box_edge_counts_as_overlapping(self):
        m = rect(60, 240, 60, 200) | rect(60, 240, 240, 340)
        _, stats = s.clean(m, [300, 100, 500, 200])  # box only clips the right blob
        self.assertEqual(stats["components_merged"], 1)

    def test_large_component_wholly_outside_box_is_rejected(self):
        m = rect(60, 240, 60, 200) | rect(60, 240, 240, 340)
        out, stats = s.clean(m, [50, 50, 210, 250])  # box covers the left blob only
        self.assertEqual(stats["components_merged"], 0)
        self.assertFalse(out[150, 300])
        self.assertAlmostEqual(stats["discarded_area_frac"], 18000 / (25200 + 18000), places=4)

    def test_close_seals_a_hairline_split_but_not_a_wide_gap(self):
        split = rect(60, 240, 60, 200) | rect(60, 240, 203, 340)  # 3 px gap, < 5 px kernel
        out, _ = s.clean(split, DINO_BOX)
        self.assertEqual(components(out), 1)
        out, _ = s.clean(TWO_BLOB, DINO_BOX)  # 40 px gap: merged into one mask, still two blobs
        self.assertEqual(components(out), 2)

    def test_windows_filled_but_border_notch_left_open(self):
        m = GOOD.copy()
        m[100:110, 120:130] = False  # enclosed window
        m[200:240, 200:220] = False  # notch open to the bottom of the building, not the frame
        m[60:70, 150:160] = False  # notch open at the roofline
        out, _ = s.clean(m, DINO_BOX)
        self.assertTrue(out[100:110, 120:130].all())
        self.assertFalse(out[65, 155])

    def test_empty_in_empty_out(self):
        out, stats = s.clean(np.zeros((H, W), bool), DINO_BOX)
        self.assertFalse(out.any())
        self.assertEqual(stats, {"components_total": 0, "components_merged": 0, "discarded_area_frac": 0.0})


class SegmentImageTests(unittest.TestCase):
    def setUp(self):
        self.image = Image.new("RGB", (W, H))
        patcher = mock.patch.object(s, "detect_boxes", return_value=DETECTIONS)
        patcher.start()
        self.addCleanup(patcher.stop)
        stderr = io.StringIO()
        redirect = contextlib.redirect_stderr(stderr)
        redirect.__enter__()
        self.addCleanup(redirect.__exit__, None, None, None)
        self.stderr = stderr

    def run_with(self, *sam_masks):
        session, calls = fake_sam(*sam_masks)
        with mock.patch.object(s, "sam_session", session):
            mask, meta = s.segment_image(self.image)
        return mask, meta, calls

    def test_good_mask_needs_no_retry(self):
        mask, meta, calls = self.run_with(GOOD)
        self.assertEqual(meta["prompt_used"], "dino_box")
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0], (DINO_BOX, None))
        self.assertEqual((meta["components_merged"], meta["discarded_area_frac"]), (0, 0.0))

    def test_mask_touching_four_edges_is_reprompted_with_centre_box(self):
        _, meta, calls = self.run_with(FOUR_EDGES, GOOD)
        self.assertEqual(meta["prompt_used"], "centre_box_retry")
        first = meta["attempts"][0]["problems"]
        self.assertEqual(len(first), 1)
        self.assertIn("4 image edges", first[0])
        box, point = calls[1]
        np.testing.assert_allclose(box, CENTRE_BOX)
        self.assertEqual(point, (W / 2, H / 2))
        self.assertIn("rejected", self.stderr.getvalue())

    def test_mask_covering_95_percent_is_reprompted(self):
        self.assertGreater(COVER_95.mean(), 0.95)
        _, meta, _ = self.run_with(COVER_95, GOOD)
        self.assertEqual(meta["prompt_used"], "centre_box_retry")
        first = meta["attempts"][0]["problems"]
        self.assertEqual(len(first), 1)
        self.assertIn("covers", first[0])

    def test_two_blob_mask_is_merged_and_recorded(self):
        mask, meta, calls = self.run_with(TWO_BLOB)
        self.assertEqual(len(calls), 1)  # accepted first time
        self.assertEqual((meta["components_total"], meta["components_merged"]), (2, 1))
        self.assertEqual(meta["discarded_area_frac"], 0.0)
        self.assertTrue(mask[150, 100] and mask[150, 300])

    def test_discard_warning_fires_above_15_percent(self):
        # Three 3,400 px blobs, each 7.9% of GOOD (43,200 px) so none merges: 19% of the mask is discarded.
        m = GOOD | rect(0, 50, 0, 68) | rect(250, 300, 0, 68) | rect(250, 300, 332, 400)
        _, meta, _ = self.run_with(m)
        self.assertEqual(meta["components_merged"], 0)
        self.assertGreater(meta["discarded_area_frac"], s.DISCARDED_WARN_FRACTION)
        self.assertIn("WARNING clean() discarded", self.stderr.getvalue())

    def test_rejected_twice_raises_with_both_attempts_listed(self):
        session, calls = fake_sam(FOUR_EDGES, COVER_95)
        with mock.patch.object(s, "sam_session", session):
            with self.assertRaises(s.SegmentationError) as ctx:
                s.segment_image(self.image)
        self.assertEqual(len(calls), 2)
        msg = str(ctx.exception)
        self.assertIn("[dino_box]", msg)
        self.assertIn("[centre_box_retry]", msg)
        self.assertIn("4 image edges", msg)
        self.assertIn("covers", msg)

    def test_no_detection_raises_before_sam_is_touched(self):
        with mock.patch.object(s, "detect_boxes", return_value=[]):
            with mock.patch.object(s, "sam_session", side_effect=AssertionError("SAM must not load")):
                with self.assertRaisesRegex(s.SegmentationError, "found nothing"):
                    s.segment_image(self.image)


class CommandLineTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)
        self.photo = self.tmp / "photo.jpg"
        Image.new("RGB", (W, H), (90, 120, 150)).save(self.photo)
        self.out = self.tmp / "out" / "mask.png"
        for name, value in (("CACHE_DIR", self.tmp / "cache"), ("_device", lambda: "cpu"), ("detect_boxes", lambda image: DETECTIONS)):
            patcher = mock.patch.object(s, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def main(self, sam_session):
        stderr = io.StringIO()
        with mock.patch.object(s, "sam_session", sam_session), mock.patch.object(
            sys, "argv", ["segment.py", str(self.photo), "-o", str(self.out)]
        ), contextlib.redirect_stderr(stderr), contextlib.redirect_stdout(io.StringIO()):
            code = s.main()
        return code, stderr.getvalue()

    def test_exit_1_and_nothing_written_when_mask_is_rejected_twice(self):
        session, _ = fake_sam(FOUR_EDGES, COVER_95)
        code, err = self.main(session)
        self.assertEqual(code, 1)
        self.assertIn("error: mask rejected after re-prompting", err)
        self.assertFalse(self.out.exists())
        self.assertFalse((self.tmp / "cache").exists())

    def test_exit_1_on_missing_file(self):
        self.photo.unlink()
        session, _ = fake_sam()
        code, err = self.main(session)
        self.assertEqual(code, 1)
        self.assertIn("no such file", err)

    def test_retry_success_writes_png_sidecar_and_is_then_cached(self):
        session, _ = fake_sam(FOUR_EDGES, TWO_BLOB)
        code, _ = self.main(session)
        self.assertEqual(code, 0)
        png = np.array(Image.open(self.out))
        self.assertEqual(png.shape, (H, W))
        self.assertEqual(set(np.unique(png)), {0, 255})
        sidecar = json.loads(next((self.tmp / "cache").glob("*.json")).read_text())
        self.assertEqual(sidecar["prompt_used"], "centre_box_retry")
        self.assertEqual(sidecar["components_merged"], 1)
        self.assertEqual(sidecar["merge_min_fraction"], s.MERGE_MIN_FRACTION)
        self.assertEqual(sidecar["discarded_area_frac"], 0.0)
        first_bytes = self.out.read_bytes()
        self.out.unlink()
        # Second run: same image bytes -> cache hit; a session that fails if touched proves no model ran.
        code, err = self.main(mock.Mock(side_effect=AssertionError("cache hit must not load SAM")))
        self.assertEqual(code, 0)
        self.assertIn("cache hit", err)
        self.assertEqual(self.out.read_bytes(), first_bytes)


if __name__ == "__main__":
    unittest.main()
