"""The rebuilt model called a real conference photo "ai_edited" at 0.82 with
the mask on a glossy tablecloth. Two answers to that live here: an abstain
rule so the UI can say "cannot tell" instead of guessing, and a training
augmentation that shows the decoder smooth patches of real photos with a
zero mask target so smoothness alone stops being an edit cue.

These tests pin the properties that matter, not the numbers: the rule
abstains exactly when the top probability is under the line, the real-image
augmentation changes pixels but never the label or mask, and the edited-image
augmentation touches only the inpainted region.
"""
import os
import sys
import unittest

import numpy as np
from PIL import Image

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for p in (ROOT, os.path.join(ROOT, "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

from frontend.mask_head import ABSTAIN, apply_rule, load_decision_rule, DEFAULT_ABSTAIN_BELOW
from training.train_mask_head import noise_in_mask_augment, smooth_patch_augment


class TestAbstainRule(unittest.TestCase):
    def test_answers_when_top_probability_clears_the_line(self):
        self.assertEqual(apply_rule({"Real": 0.95, "AI Generated": 0.03, "AI Edited": 0.02}, 0.9), "Real")

    def test_abstains_just_under_the_line(self):
        # the photo that motivated this scored 0.82 on the wrong class
        self.assertEqual(apply_rule({"Real": 0.178, "AI Generated": 0.0, "AI Edited": 0.821}, 0.9), ABSTAIN)

    def test_line_is_inclusive(self):
        self.assertEqual(apply_rule({"Real": 0.9, "AI Generated": 0.05, "AI Edited": 0.05}, 0.9), "Real")

    def test_zero_line_never_abstains(self):
        self.assertEqual(apply_rule({"Real": 0.34, "AI Generated": 0.33, "AI Edited": 0.33}, 0.0), "Real")

    def test_missing_rule_file_falls_back_to_default(self):
        rule = load_decision_rule(os.path.join(ROOT, "nonexistent", "best_model.pth"))
        self.assertEqual(rule["abstain_below"], DEFAULT_ABSTAIN_BELOW)


def _photo(seed=0, size=(96, 64)):
    rng = np.random.default_rng(seed)
    # textured, so smoothing has something to remove
    a = rng.integers(0, 256, (size[1], size[0], 3), dtype=np.uint8)
    return Image.fromarray(a, "RGB")


class TestSmoothPatchAugment(unittest.TestCase):
    def test_changes_pixels_but_not_size_or_mode(self):
        img = _photo()
        out = smooth_patch_augment(img, np.random.default_rng(1))
        self.assertEqual(out.size, img.size)
        self.assertEqual(out.mode, "RGB")
        self.assertGreater(np.abs(np.asarray(out, float) - np.asarray(img, float)).mean(), 0.5)

    def test_does_not_touch_the_input(self):
        img = _photo(); before = np.asarray(img).copy()
        smooth_patch_augment(img, np.random.default_rng(2))
        self.assertTrue((np.asarray(img) == before).all())

    def test_only_part_of_the_image_changes(self):
        # regions cover 3 to 25 percent each, up to three, so most pixels stay
        img = _photo(size=(200, 200))
        out = smooth_patch_augment(img, np.random.default_rng(3))
        changed = (np.abs(np.asarray(out, int) - np.asarray(img, int)).sum(-1) > 0).mean()
        self.assertLess(changed, 0.8)
        self.assertGreater(changed, 0.0)

    def test_deterministic_for_a_seed(self):
        img = _photo()
        a = smooth_patch_augment(img, np.random.default_rng(7))
        b = smooth_patch_augment(img, np.random.default_rng(7))
        self.assertTrue((np.asarray(a) == np.asarray(b)).all())


class TestNoiseInMaskAugment(unittest.TestCase):
    def test_noise_lands_only_inside_the_mask(self):
        img = _photo(size=(64, 64))
        mask = Image.new("L", (64, 64), 0)
        mask.paste(255, (16, 16, 48, 48))
        out = noise_in_mask_augment(img, mask, np.random.default_rng(0))
        diff = np.abs(np.asarray(out, int) - np.asarray(img, int)).sum(-1)
        m = np.asarray(mask) > 127
        self.assertEqual(diff[~m].max(), 0)
        self.assertGreater(diff[m].mean(), 0.5)

    def test_mask_at_a_different_resolution_is_resized(self):
        img = _photo(size=(64, 64))
        mask = Image.new("L", (32, 32), 0); mask.paste(255, (8, 8, 24, 24))
        out = noise_in_mask_augment(img, mask, np.random.default_rng(0))
        diff = np.abs(np.asarray(out, int) - np.asarray(img, int)).sum(-1)
        self.assertEqual(diff[:16, :].max(), 0)     # outside the upscaled mask
        self.assertGreater(diff[16:48, 16:48].mean(), 0.5)


if __name__ == "__main__":
    unittest.main()
