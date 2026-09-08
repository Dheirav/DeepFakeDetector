"""SRM was switched off at initialisation, and FFT depended on batch composition.

adapt_conv1_for_srm tiled the RED channel across all three residual channels, so
they began bit-identical -- collapsing three differently-oriented kernels into
their sum -- at roughly 1/1250 of the pre-activation variance. Reaching parity
needed ~35x weight growth, which does not happen at lr=1e-4 over 11-30 epochs.
The reported 'SRM makes no difference' was therefore a finding about the
initialiser, not about SRM.

FFTLayer normalised with bare .min()/.max(), which reduce over the batch
dimension too, so an image's features depended on its batch-mates.
"""
import unittest

import torch

from preprocessing.fft import FFTLayer
from preprocessing.srm import SRMLayer, adapt_conv1_for_srm


class TestSRMLayer(unittest.TestCase):
    def test_rgb_channels_pass_through_untouched(self):
        x = torch.randn(2, 3, 32, 32)
        out = SRMLayer()(x)
        self.assertEqual(out.shape, (2, 6, 32, 32))
        self.assertTrue(torch.equal(out[:, :3], x))

    def test_residual_channels_differ_from_each_other(self):
        out = SRMLayer()(torch.randn(1, 3, 32, 32))
        self.assertFalse(torch.equal(out[:, 3], out[:, 4]))
        self.assertFalse(torch.equal(out[:, 4], out[:, 5]))


class TestSRMInit(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.conv = adapt_conv1_for_srm(torch.nn.Conv2d(3, 64, 7, bias=False), 6)

    def test_residual_channels_are_not_identical_copies(self):
        w = self.conv.weight
        self.assertFalse(torch.equal(w[:, 3], w[:, 4]))
        self.assertFalse(torch.equal(w[:, 4], w[:, 5]))

    def test_residual_branch_is_not_negligible(self):
        """It contributed ~0.0008 of the pre-activation variance before."""
        w = self.conv.weight
        share = (w[:, 3:].pow(2).sum() * 0.28 ** 2 / w[:, :3].pow(2).sum()).item()
        self.assertGreater(share, 0.01, "residual branch is effectively switched off")

    def test_pretrained_rgb_weights_are_preserved(self):
        src = torch.nn.Conv2d(3, 64, 7, bias=False)
        self.assertTrue(torch.equal(adapt_conv1_for_srm(src, 6).weight[:, :3], src.weight))

    def test_no_op_when_channel_count_already_matches(self):
        src = torch.nn.Conv2d(6, 64, 7, bias=False)
        self.assertIs(adapt_conv1_for_srm(src, 6), src)


class TestFFTLayer(unittest.TestCase):
    def test_output_is_independent_of_batch_composition(self):
        torch.manual_seed(0)
        target = torch.randn(1, 3, 64, 64)
        alone = FFTLayer()(target)
        for batch in (4, 16):
            with self.subTest(batch_size=batch):
                others = torch.randn(batch - 1, 3, 64, 64) * 5.0
                together = FFTLayer()(torch.cat([target, others]))[:1]
                self.assertTrue(torch.allclose(alone, together, atol=1e-5),
                                f"features shifted by {(alone - together).abs().max():.4f} "
                                f"at batch_size={batch}")

    def test_each_sample_is_normalised_to_the_unit_range(self):
        out = FFTLayer()(torch.randn(4, 3, 64, 64))
        self.assertEqual(out.shape, (4, 1, 64, 64))
        for i in range(4):
            self.assertAlmostEqual(out[i].min().item(), 0.0, places=4)
            self.assertAlmostEqual(out[i].max().item(), 1.0, places=4)


if __name__ == "__main__":
    unittest.main()
