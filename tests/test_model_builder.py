"""Loaders inferred the architecture from state-dict key names and got it wrong.

Two published numbers described models that were never trained:

  results/20  FFTLayer has no parameters, so no key name mentions it. A bare
              backbone was built and ZERO weights loaded (missing=344,
              unexpected=344). Reported 34.91% -- chance -- against an 88.37%
              training log.
  results/18  CBAM re-nests `features`, so the ConvNeXt-Small probe
              ('features.5.20') missed and Small loaded as Tiny. Identical
              channel widths meant no size mismatch, so strict=False silently
              dropped 162 tensors. Reported 70.94% against 83.56%.

Both were invisible because loading was tolerant. These tests exist to keep it
intolerant.
"""
import json
import os
import tempfile
import unittest

import torch

from modules.model_builder import build_model, config_for_checkpoint, load_model

CONFIGS = [
    {"backbone": "resnet18", "num_classes": 3, "dropout_p": 0.4,
     "use_srm": False, "use_fft": False, "attention_head": "none"},
    {"backbone": "resnet18", "num_classes": 3, "dropout_p": 0.4,
     "use_srm": True, "use_fft": False, "attention_head": "none"},
    {"backbone": "resnet18", "num_classes": 3, "dropout_p": 0.0,
     "use_srm": False, "use_fft": True, "attention_head": "none"},
    {"backbone": "resnet18", "num_classes": 2, "dropout_p": 0.4,
     "use_srm": True, "use_fft": True, "attention_head": "gem"},
    {"backbone": "resnet18", "num_classes": 3, "dropout_p": 0.4,
     "use_srm": True, "use_fft": False, "attention_head": "cbam"},
]


class TestRoundTrip(unittest.TestCase):
    def test_every_config_saves_and_reloads_identically(self):
        for cfg in CONFIGS:
            with self.subTest(cfg=cfg):
                model = build_model(cfg)
                with tempfile.TemporaryDirectory() as d:
                    path = os.path.join(d, "ckpt.pth")
                    torch.save({"state_dict": model.state_dict(), "config": cfg}, path)
                    loaded, back = load_model(path, device="cpu")
                self.assertEqual(back["backbone"], cfg["backbone"])
                for a, b in zip(model.state_dict().values(), loaded.state_dict().values()):
                    self.assertTrue(torch.equal(a, b))

    def test_input_and_output_shapes_follow_the_config(self):
        for cfg in CONFIGS:
            with self.subTest(cfg=cfg):
                out = build_model(cfg).eval()(torch.randn(2, 3, 64, 64))
                self.assertEqual(out.shape, (2, cfg["num_classes"]))


class TestStrictness(unittest.TestCase):
    def test_a_mismatched_checkpoint_is_rejected_not_silently_accepted(self):
        """This is the results/18 and results/20 failure mode."""
        srm = {"backbone": "resnet18", "num_classes": 3, "dropout_p": 0.4,
               "use_srm": True, "use_fft": False, "attention_head": "none"}
        plain = dict(srm, use_srm=False)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "ckpt.pth")
            torch.save({"state_dict": build_model(srm).state_dict(), "config": srm}, path)
            with self.assertRaises(RuntimeError):
                load_model(path, device="cpu", cfg=plain)

    def test_refuses_to_guess_when_no_config_exists(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "orphan.pth")
            torch.save(build_model(CONFIGS[0]).state_dict(), path)
            with self.assertRaises(FileNotFoundError):
                config_for_checkpoint(path)

    def test_compiled_checkpoints_load(self):
        """torch.compile prefixes every key with _orig_mod."""
        cfg = CONFIGS[0]
        model = build_model(cfg)
        state = {f"_orig_mod.{k}": v for k, v in model.state_dict().items()}
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "compiled.pth")
            torch.save({"state_dict": state, "config": cfg}, path)
            loaded, _ = load_model(path, device="cpu")
        for a, b in zip(model.state_dict().values(), loaded.state_dict().values()):
            self.assertTrue(torch.equal(a, b))

    def test_a_config_without_a_backbone_raises(self):
        with self.assertRaises(ValueError):
            build_model({"num_classes": 3})


if __name__ == "__main__":
    unittest.main()
