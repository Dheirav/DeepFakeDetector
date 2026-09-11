# CLIP ViT-B/16 @448, last 4 encoder blocks fine-tuned

Identical to `../mask_head_clip448_balanced/` in every argument except
`--unfreeze 4` (encoder blocks 8 to 11 and `ln_post` trained at 0.1x the head's
learning rate). Best epoch 4 of 10.

    in-distribution (same 4,050 images as the frozen run)
        accuracy 0.9015   real F1 0.858   ai_generated 0.990   ai_edited 0.855   IoU 0.401
        McNemar vs frozen 0.8040: 555 vs 160 discordant, p = 3.9e-49

    held-out generators, ai_generated recall      frozen -> fine-tuned
        sd2 0.820 -> 0.850   sd3 0.660 -> 0.505   sdxl 0.635 -> 0.328   flux 0.647 -> 0.240
    held-out generators, ai_edited recall
        sd15 0.715 -> 0.853  sd3 0.765 -> 0.797   sdxl 0.690 -> 0.718   flux 0.642 -> 0.657
    mean held-out recall over 9 rows: 0.723 -> 0.657

It is the better model on the test split and the worse detector on anything
not made by sd15. Read `train.log` for the per-epoch curve and
`heldout_eval.log` for the full held-out table.

`best_model.pth` (116 MB, carries the 28.4M fine-tuned encoder parameters) is
not in git because it exceeds GitHub's per-file limit. A copy with its sha256
is at `/mnt/c/d_drive/projects/deepfake_models/mask_head_clip448_ft4/` on the
Windows side. The frozen-encoder checkpoints are all committed.
