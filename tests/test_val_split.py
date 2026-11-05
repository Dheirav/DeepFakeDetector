"""The validation split must never move the test split.

Every mask-head run is scored on the same stratified 30 percent test split, which
is what makes McNemar valid across runs. Asking for a validation split carves
it out of the training indices only. These tests pin that, plus the properties
that make the split usable: disjoint, covering, stratified, deterministic.
"""
import os
import sys
import unittest

import numpy as np
from sklearn.model_selection import train_test_split

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for p in (ROOT, os.path.join(ROOT, "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

from training.train_mask_head import make_splits


def labels(n_per=300):
    return np.repeat([0, 1, 2], n_per)


class MakeSplitsTest(unittest.TestCase):
    def test_no_val_matches_the_original_single_split(self):
        y = labels()
        tr, va, te = make_splits(y, 0.3, 0.0, 42)
        otr, ote = train_test_split(np.arange(len(y)), test_size=0.3,
                                    random_state=42, stratify=y)
        np.testing.assert_array_equal(tr, otr)
        np.testing.assert_array_equal(te, ote)
        self.assertEqual(len(va), 0)

    def test_test_split_is_identical_with_and_without_val(self):
        y = labels()
        _, _, te0 = make_splits(y, 0.3, 0.0, 42)
        _, _, te1 = make_splits(y, 0.3, 0.15, 42)
        np.testing.assert_array_equal(te0, te1)

    def test_val_comes_out_of_train_only(self):
        y = labels()
        tr0, _, te = make_splits(y, 0.3, 0.0, 42)
        tr, va, _ = make_splits(y, 0.3, 0.15, 42)
        self.assertEqual(set(tr) | set(va), set(tr0))
        self.assertFalse(set(tr) & set(va))
        self.assertFalse((set(tr) | set(va)) & set(te))

    def test_val_is_stratified_and_sized(self):
        y = labels()
        tr, va, _ = make_splits(y, 0.3, 0.15, 42)
        counts = np.bincount(y[va], minlength=3)
        # Stratified rounding can differ by one image between classes.
        self.assertLessEqual(counts.max() - counts.min(), 1)
        self.assertAlmostEqual(len(va) / (len(tr) + len(va)), 0.15, delta=0.01)

    def test_deterministic_and_seed_sensitive(self):
        y = labels()
        a = make_splits(y, 0.3, 0.15, 42)
        b = make_splits(y, 0.3, 0.15, 42)
        for u, v in zip(a, b):
            np.testing.assert_array_equal(u, v)
        c = make_splits(y, 0.3, 0.15, 7)
        self.assertFalse(np.array_equal(a[2], c[2]))


if __name__ == "__main__":
    unittest.main()
