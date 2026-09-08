"""Every Grad-CAM figure in this project was colour-inverted.

cv2.applyColorMap returns BGR; the photo is RGB. Blending them without
converting flipped the colormap end for end, so the MOST important regions
rendered blue and the least important red -- while the UI told the reader the
opposite in words. Because jet is roughly red/blue symmetric in the mid-range,
an inverted heatmap still looks like a plausible heatmap, which is why it
survived unnoticed.

The target layer was wrong too: taking the last nn.Conv2d in module order lands
on CBAM's spatial-attention conv, which has ONE output channel, making the CAM
class-independent and therefore meaningless.
"""
import os
import unittest

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as tvm
from PIL import Image

from frontend.gradcam import GradCAM, _overlay_matplotlib, _overlay_opencv
from modules.attention_heads import CBAMBlock

GREY = Image.new("RGB", (32, 32), (128, 128, 128))
HOT = np.ones((32, 32), dtype=np.float32)
COLD = np.zeros((32, 32), dtype=np.float32)


def centre(img):
    return np.array(img)[16, 16]


class TestOverlayOrientation(unittest.TestCase):
    def test_high_importance_renders_warm_on_both_paths(self):
        for name, fn in (("opencv", _overlay_opencv), ("matplotlib", _overlay_matplotlib)):
            with self.subTest(path=name):
                hot, cold = centre(fn(GREY, HOT, 1.0, "jet")), centre(fn(GREY, COLD, 1.0, "jet"))
                self.assertGreater(hot[0], hot[2], f"{name}: importance 1.0 should be warm")
                self.assertGreater(cold[2], cold[0], f"{name}: importance 0.0 should be cool")

    def test_both_render_paths_agree(self):
        a, b = centre(_overlay_opencv(GREY, HOT, 1.0, "jet")), centre(_overlay_matplotlib(GREY, HOT, 1.0, "jet"))
        self.assertEqual(a[0] > a[2], b[0] > b[2])


class TestTargetLayer(unittest.TestCase):
    def _pick(self, model):
        cam = GradCAM.__new__(GradCAM)
        cam.backbone = model
        return cam._find_target_module(model)

    def test_never_selects_a_single_channel_attention_conv(self):
        for name, model in (("resnet50+cbam", tvm.resnet50(weights=None)),
                            ("convnext+cbam", tvm.convnext_small(weights=None))):
            with self.subTest(model=name):
                if hasattr(model, "layer4"):
                    model.layer4 = nn.Sequential(model.layer4, CBAMBlock(channels=2048))
                else:
                    model.features = nn.Sequential(model.features, CBAMBlock(channels=768))
                picked = self._pick(model)
                self.assertNotIsInstance(picked, nn.Conv2d)

    def test_picks_the_stage_container_for_plain_backbones(self):
        """The CAM is defined on the last stage's OUTPUT, not on some conv inside
        it. The old rule took the last nn.Conv2d in module order, which for
        ConvNeXt is the depthwise conv at the *entrance* of the final block --
        before LayerNorm, MLP, layer-scale and the residual add (Pearson r=0.598
        against the canonical CAM) -- and for ResNet a 1x1 conv before BN and
        the residual add."""
        resnet = tvm.resnet18(weights=None)
        self.assertIs(self._pick(resnet), resnet.layer4)

        convnext = tvm.convnext_tiny(weights=None)
        self.assertIs(self._pick(convnext), convnext.features)


class TestCamOutput(unittest.TestCase):
    """Grad-CAM applies ReLU to the weighted activation sum, so an untrained
    network on random input can legitimately produce an all-zero map. These
    checks therefore need real weights, and are skipped when none are present --
    models/ is gitignored, so CI will skip them."""

    CKPT = "models/19__convnext-small__light__0.4__cosine__focal__none/best_model.pth"

    def setUp(self):
        if not os.path.isfile(self.CKPT):
            self.skipTest(f"no checkpoint at {self.CKPT}")
        from modules.model_builder import load_model
        self.model, _ = load_model(self.CKPT, device="cpu")

    def test_cam_is_class_dependent_and_not_degenerate(self):
        cam = GradCAM(self.model)
        try:
            maps = [cam(torch.randn(1, 3, 224, 224), class_idx=c) for c in range(3)]
        finally:
            cam.cleanup()
        for c, m in enumerate(maps):
            self.assertGreater(float(m.max() - m.min()), 0.0, f"class {c}: CAM is uniform")
        r = np.corrcoef(maps[0].ravel(), maps[1].ravel())[0, 1]
        self.assertLess(abs(r), 0.999, "CAM does not depend on the target class")


class TestCamShape(unittest.TestCase):
    def test_hooks_fire_and_a_map_of_the_right_shape_comes_back(self):
        """Shape and hook wiring do not need trained weights."""
        model = tvm.resnet18(weights=None)
        model.fc = nn.Linear(512, 3)
        cam = GradCAM(model.eval())
        try:
            out = cam(torch.randn(1, 3, 224, 224), class_idx=0)
            # Read these before cleanup(), which nulls them by design.
            acts, grads = cam.activations, cam.gradients
        finally:
            cam.cleanup()
        self.assertEqual(out.ndim, 2)
        self.assertIsNotNone(acts, "forward hook did not fire")
        self.assertIsNotNone(grads, "backward hook did not fire")
        self.assertTrue(bool((grads != 0).any()), "gradients are all zero")


if __name__ == "__main__":
    unittest.main()
