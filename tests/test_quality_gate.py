"""The input quality gate reads the raw upload, so the properties that matter
are: it recovers the JPEG quality the file was written at, it does not invent
one for files it cannot read, and the bands it produces sort the measured
degradation grid the way the grid sorts itself (ok above 0.9, warn between,
poor below 0.8)."""
import io
import os
import sys
import unittest

import numpy as np
from PIL import Image

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from frontend import quality as Q


def jpeg_bytes(q, size=(64, 48)):
    rng = np.random.default_rng(0)
    im = Image.fromarray(rng.integers(0, 255, (size[1], size[0], 3), dtype=np.uint8))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=q)
    return buf.getvalue()


class EstimateTest(unittest.TestCase):
    def test_recovers_written_quality(self):
        for q in (30, 50, 60, 75, 85, 90, 95):
            im = Image.open(io.BytesIO(jpeg_bytes(q)))
            self.assertLessEqual(abs(Q.estimate_jpeg_quality(im.quantization) - q), 1, q)

    def test_png_and_missing_tables_give_none(self):
        buf = io.BytesIO()
        Image.new("RGB", (600, 500)).save(buf, "PNG")
        buf.seek(0)
        self.assertEqual(Q.read_source(buf), (500, None))
        self.assertIsNone(Q.estimate_jpeg_quality(None))
        self.assertIsNone(Q.estimate_jpeg_quality({}))

    def test_nonstandard_table_gives_none_not_a_guess(self):
        odd = {0: [7 if i % 2 else 200 for i in range(64)]}
        self.assertIsNone(Q.estimate_jpeg_quality(odd))

    def test_read_source_rewinds_and_reports_short_side(self):
        f = io.BytesIO(jpeg_bytes(90, size=(300, 200)))
        short, q = Q.read_source(f)
        self.assertEqual(short, 200)
        self.assertLessEqual(abs(q - 90), 1)
        self.assertEqual(f.tell(), 0)


class BandsTest(unittest.TestCase):
    def test_bands_sort_the_degradation_grid(self):
        # (short side, jpeg q, accuracy-when-answered from degradation.json)
        grid = [(512, 90, 0.958), (448, 90, 0.922), (384, 75, 0.901), (320, 85, 0.830),
                (256, 80, 0.846), (256, 60, 0.794), (224, 50, 0.705)]
        want = [Q.OK, Q.OK, Q.WARN, Q.WARN, Q.WARN, Q.POOR, Q.POOR]
        self.assertEqual([Q.assess(s, q).level for s, q, _ in grid], want)
        for s, q, acc in grid:
            lv = Q.assess(s, q).level
            if lv == Q.OK:
                self.assertGreater(acc, 0.9)
            elif lv == Q.POOR:
                self.assertLess(acc, 0.8)

    def test_edges(self):
        self.assertEqual(Q.assess(255).level, Q.POOR)
        self.assertEqual(Q.assess(256).level, Q.WARN)
        self.assertEqual(Q.assess(447).level, Q.WARN)
        self.assertEqual(Q.assess(448).level, Q.OK)
        self.assertEqual(Q.assess(3000, 64).level, Q.POOR)
        self.assertEqual(Q.assess(3000, 65).level, Q.WARN)
        self.assertEqual(Q.assess(3000, 85).level, Q.OK)
        self.assertEqual(Q.assess(3000, None).level, Q.OK)

    def test_worst_reason_wins_and_all_are_listed(self):
        r = Q.assess(300, 50)
        self.assertEqual(r.level, Q.POOR)
        self.assertEqual(len(r.reasons), 2)


class GateTest(unittest.TestCase):
    def test_poor_withholds_warn_annotates_ok_is_silent(self):
        v, n = Q.gate("real", Q.assess(200), "Cannot tell")
        self.assertEqual(v, "Cannot tell")
        self.assertIn("withheld", n)
        v, n = Q.gate("real", Q.assess(300), "Cannot tell")
        self.assertEqual(v, "real")
        self.assertIsNotNone(n)
        self.assertEqual(Q.gate("real", Q.assess(1000, 92), "Cannot tell"), ("real", None))


if __name__ == "__main__":
    unittest.main()
