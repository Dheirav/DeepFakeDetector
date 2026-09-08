"""FocalLoss derived p_t from the weighted, label-smoothed cross-entropy.

exp(-ce) is only the model's probability for the correct class when ce is plain
cross-entropy. With gamma=3 and label_smoothing=0.1 -- the project defaults -- a
99%-confident correct prediction was down-weighted by 0.049 instead of 1e-6, so
the focusing mechanism the class exists to provide was effectively off. The
class weight also leaked into the modulator, making it class-dependent and
inverting the intended emphasis at high confidence.
"""
import unittest

import torch
import torch.nn.functional as F

from training.losses import FocalLoss, build_criterion


def logits_for(p, target, n=3):
    """Logits giving probability p to `target` and (1-p)/(n-1) to each other."""
    rest = (1 - p) / (n - 1)
    row = [rest] * n
    row[target] = p
    return torch.log(torch.tensor([row]))


class TestFocalModulation(unittest.TestCase):
    def test_modulator_equals_one_minus_p_to_the_gamma(self):
        weight = torch.tensor([1.5, 1.0, 1.5])
        for gamma in (2.0, 3.0):
            for smoothing in (0.0, 0.1):
                crit = FocalLoss(gamma=gamma, weight=weight,
                                 label_smoothing=smoothing, reduction="none")
                for p in (0.99, 0.9, 0.7, 0.5):
                    with self.subTest(gamma=gamma, ls=smoothing, p=p):
                        lg, t = logits_for(p, 0), torch.tensor([0])
                        ce = F.cross_entropy(lg, t, weight=weight, reduction="none",
                                             label_smoothing=smoothing)
                        got = (crit(lg, t) / ce).item()
                        self.assertAlmostEqual(got, (1 - p) ** gamma, places=5)

    def test_modulator_does_not_depend_on_the_class_weight(self):
        """At p=0.99 the weighted implementation gave Real (w=1.5) a *smaller*
        modulator than AI-Generated (w=1.0) -- the opposite of the intent."""
        crit = FocalLoss(gamma=3.0, weight=torch.tensor([1.5, 1.0, 1.5]),
                         label_smoothing=0.1, reduction="none")
        mods = []
        for cls in (0, 1):
            lg, t = logits_for(0.99, cls), torch.tensor([cls])
            ce = F.cross_entropy(lg, t, weight=torch.tensor([1.5, 1.0, 1.5]),
                                 reduction="none", label_smoothing=0.1)
            mods.append((crit(lg, t) / ce).item())
        self.assertAlmostEqual(mods[0], mods[1], places=6)

    def test_gamma_zero_is_plain_cross_entropy(self):
        crit = FocalLoss(gamma=0.0, reduction="none")
        lg, t = logits_for(0.6, 1), torch.tensor([1])
        self.assertAlmostEqual(crit(lg, t).item(),
                               F.cross_entropy(lg, t, reduction="none").item(), places=6)


class TestBuildCriterion(unittest.TestCase):
    def test_every_loss_type_builds_and_returns_a_scalar(self):
        lg = torch.randn(4, 3)
        t = torch.tensor([0, 1, 2, 0])
        for name in ("ce", "weighted", "focal", "weighted_focal"):
            with self.subTest(loss=name):
                out = build_criterion(name, "cpu", 0.1)(lg, t)
                self.assertEqual(out.shape, torch.Size([]))
                self.assertTrue(torch.isfinite(out))

    def test_unknown_loss_type_raises(self):
        with self.assertRaises(ValueError):
            build_criterion("nonsense", "cpu")


if __name__ == "__main__":
    unittest.main()
