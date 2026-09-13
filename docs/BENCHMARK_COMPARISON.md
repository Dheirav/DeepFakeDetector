# How our results compare to the OpenSDI paper

Compared against Wang et al., *OpenSDI: Spotting Diffusion-Generated Images in
the Open World*, CVPR 2025 ([arXiv:2503.19653](https://arxiv.org/abs/2503.19653)),
which is the paper that introduced the dataset we train on.

## What is not comparable, stated first

Four differences matter, and three of them favour the paper heavily:

| | paper | ours |
|---|---|---|
| task | **binary**, real vs fake | **3-class**, real / generated / edited |
| training images | **200,000** | 13,500 balanced, or 14,822 with real tripled (**7%**) |
| encoder | fully trained | **frozen**, only a small decoder and head train |
| resolution | 224 detect, 512 localise | 448 both |

Our three-class task is strictly harder than their binary one, so for any
comparison against their numbers we collapse ours to real-vs-fake. We also report
balanced accuracy throughout. The earlier CLIP run had a test split that was 58%
real, where raw accuracy is not comparable to a 50/50 set; the current run trains
and tests on 4,500 per class, so raw and balanced coincide and no prior
correction is needed.

## Detection

Balanced real-vs-fake accuracy, in-domain on SD1.5:

| method | accuracy | notes |
|---|---|---|
| MaskCLIP (paper's method) | **0.9272** | 200K training images, encoder trained |
| RINE | 0.9098 | |
| **ours, CLIP ViT-B/16 @448, last 4 blocks trained** | 0.8926 | 13.5K images; see transfer below |
| **ours, CLIP ViT-B/16 @448, frozen** | **0.7948 to 0.8092** | 13.5K to 14.8K images, encoder frozen |
| IML-ViT | 0.7573 | |

The fine-tuned row sits 3.5 points under MaskCLIP on 7% of its data, which says
the in-domain gap is mostly the frozen encoder rather than data volume. It is
not the headline row, for the reason in the transfer section.

The range is two training mixes of the same model: 0.7948 trained balanced, 0.8092
with real tripled to 12,175. More real images help the binary real-vs-fake
collapse a little because that is the class they add, while for the three-class
task the same change was within noise (0.7969 prior-corrected against 0.8040
balanced). Either way we sit 12 to 13 points behind the state of the art and
above IML-ViT, which is a published baseline in the same table. Given 7% of the
training data and a frozen backbone, that is a reasonable place to be rather than
a disappointing one.

## Localisation

Mask IoU, in-domain on SD1.5:

| method | IoU |
|---|---|
| MaskCLIP | **0.6712** |
| IML-ViT | 0.6651 |
| TruFor | 0.6342 |
| **ours, DINOv2 mask head (best IoU)** | **0.385** |
| ours, CLIP mask head, last 4 blocks trained | 0.401 |
| **ours, CLIP mask head (final model)** | **0.272** |

This is the honest weak point and the gap is large. The two rows are the same
decoder on two encoders trained on the same data: the one that classifies best
localises worst, so the final model is chosen on classification and gives up
0.11 IoU for it. They train the full network at
512px on 200,000 images; we train a four-block decoder on frozen features with
13,500. Localisation is the task that benefits most from both, so the ordering is
expected, but the size of the gap is not something to explain away.

## The finding that reproduces independently

Both their results and ours show transfer collapsing with architectural distance
from the training generator, and localisation collapsing far faster than
classification.

Theirs, MaskCLIP trained on SD1.5:

| | SD1.5 | SD2.1 | SDXL | SD3 | Flux.1 | loss |
|---|---|---|---|---|---|---|
| detection accuracy | 0.9272 | 0.8945 | 0.8122 | 0.7801 | 0.6850 | -26% |
| localisation IoU | 0.6712 | 0.5550 | 0.3098 | 0.4375 | 0.1622 | **-76%** |

Ours, two mask heads trained on the identical balanced sd15 set:

| | sd15 | sd3 | sdxl | flux | loss |
|---|---|---|---|---|---|
| DINOv2 ai_edited recall | 0.670 | 0.583 | 0.550 | 0.505 | -25% |
| DINOv2 mask IoU | 0.353 | 0.268 | 0.155 | 0.111 | **-69%** |
| CLIP ai_edited recall | 0.715 | 0.765 | 0.690 | 0.642 | -10% |
| CLIP mask IoU | 0.251 | 0.222 | 0.161 | 0.161 | **-36%** |

The same asymmetry appears in all three: localisation loses several times more
than classification does. A detector on an unfamiliar generator still registers
that something is wrong while largely losing track of where. That is a property
of the problem rather than of either implementation, which is worth more than a
headline number.

CLIP flattens the curve on both rows, and the sd3 recall is actually above the
sd15 control, which says its ai_edited decision leans on something less tied to
one generator's fingerprint than DINOv2's does. It also starts from a lower IoU
on the control, 0.251 against 0.353, so the two encoders trade classification
for localisation rather than one dominating.

### The comparison that almost went wrong

The first CLIP versus DINOv2 cross-generator comparison put CLIP 13 to 29 points
behind on every generator including the control. That run of CLIP had been
trained with real tripled while DINOv2 had 4,500 per class, so CLIP carried a
much stronger prior toward real and the raw fake-class recall measured the prior
as much as the features. Prior correction closed most of the gap, but a 9 to 22
point deficit on ai_generated survived it and was reported as a real transfer
weakness. Retraining CLIP on the same balanced data settles it:

| generator | class | DINOv2 | CLIP | delta |
|---|---|---|---|---|
| sd15 (control) | ai_edited | 0.670 | 0.715 | +0.045 |
| flux | ai_generated | 0.565 | 0.647 | +0.083 |
| flux | ai_edited | 0.505 | 0.642 | +0.137 |
| sd2 | real | 0.730 | 0.930 | +0.200 |
| sd2 | ai_generated | 0.875 | 0.820 | -0.055 |
| sd3 | ai_generated | 0.705 | 0.660 | -0.045 |
| sd3 | ai_edited | 0.583 | 0.765 | +0.182 |
| sdxl | ai_generated | 0.672 | 0.635 | -0.037 |
| sdxl | ai_edited | 0.550 | 0.690 | +0.140 |

Mean held-out recall is 0.723 for CLIP against 0.651 for DINOv2.

### Fine-tuning reproduces the paper's flux collapse, harder

MaskCLIP, encoder trained on SD1.5, falls from 0.927 to 0.685 on Flux. Our CLIP
head with the last 4 blocks trained falls further on the same axis:

| | sd15 (in-domain) | sd2 | sd3 | sdxl | flux |
|---|---|---|---|---|---|
| frozen, `ai_generated` recall | 0.961 F1 | 0.820 | 0.660 | 0.635 | 0.647 |
| last 4 blocks trained | 0.990 F1 | 0.850 | 0.505 | 0.328 | 0.240 |

Same 13,500 images, one flag. The trained encoder is better on the training
generator and its nearest relative and worse on everything else, with the loss
growing with architectural distance. `ai_edited` recall held or improved on every
generator (sd15 0.715 to 0.853, flux 0.642 to 0.657), so the collapse is specific
to whole-image synthesis, where there is no edit boundary to anchor on. The
paper's 200,000 images buy it a gentler slope on the same curve; they do not
change the shape.

### A negative result on the way

Augmenting real training images with smoothed regions (mask target zero), to
stop the decoder reading smoothness as inpainting, lowered in-distribution
accuracy 0.8040 to 0.7911 (p = 0.025) and mean held-out recall 0.723 to 0.689.
Recorded in `results/mask_head_clip448_smoothaug/`. With frozen features the
decoder has no better cue to move to. The ai_generated
gap shrinks to 4 to 6 points on the SD family and reverses on flux, and the sd2
real row explains the remainder: DINOv2 reaches its higher ai_generated recall
partly by calling 27% of real images generated, which CLIP does not do. That is a
threshold shift seen from both sides, not a representation gap. The earlier
reading is withdrawn, and the lesson from it is kept: two models trained on
different class mixes cannot be compared on raw recall, and prior correction
narrows the confound without removing it.

## Our own progression

| configuration | 3-class balanced | binary | IoU |
|---|---|---|---|
| DINOv2 ViT-S linear probe | 0.7512 | - | - |
| DINOv2 ViT-S mask head @448, real tripled | 0.7292 | 0.7519 | 0.277 |
| DINOv2 ViT-S mask head @448, balanced | 0.7328 | 0.7256 | 0.385 |
| CLIP ViT-B/16 mask head @224 | 0.7835 | 0.7878 | 0.176 |
| CLIP ViT-B/16 mask head @448, real tripled | 0.7969 | **0.8092** | 0.237 |
| **CLIP ViT-B/16 mask head @448, balanced** | **0.8040** | 0.7948 | 0.272 |
| CLIP ViT-B/16 @448, balanced, last 4 blocks trained | 0.9015 | 0.8926 | 0.401 |
| CLIP ViT-B/16 @448, balanced, smooth-patch augmentation | 0.7911 | - | 0.265 |

The bold row is the model in the frontend. The row under it scores higher on
every in-domain column and is not chosen, because its mean held-out recall is
0.657 against 0.723. The last row is a negative result kept for the record.

The last row is the current model. It is trained on exactly the data the DINOv2
mask head saw, so the two are a paired comparison on the same 4,050 test images:
McNemar p = 2.02e-16, 0.7328 against 0.8040. Per class it is real F1 0.727,
ai_generated 0.961, ai_edited 0.722.

Both jumps came from the representation and how it is fed. Six other levers were
tested and eliminated with measurements: encoder capacity (p = 1.000), epochs
(converges at 3 to 4), data volume, decision rule (+0.0002 across a full sweep),
resolution 448 to 672 (within noise), and class balance, which moved DINOv2 by
-0.0036 prior-corrected and CLIP by +0.007 between its two rows above, in
opposite directions and both within noise.

Note that DINOv2 still holds the best IoU at 0.385 despite the worst
classification, partly because it ran on a 32x32 grid against CLIP's 28x28. Grid
resolution matters for localisation independently of feature quality.

## What the gap says to do next

**For detection**, the gap is mostly data and a frozen encoder. Scaling toward the
full 200,000 images is the obvious move and OpenSDI has them.

**For localisation**, the gap is larger and points at the same two causes plus
resolution. The paper localises at 512px with a fully trained network. Our 0.272
comes from a small decoder on frozen features, and an unfinished ViT-L/14 run at a
32x32 grid reached IoU 0.2545 after a single epoch, above the ViT-B/16 real-tripled run's best of
0.237 across all ten of its epochs, so there is clearly headroom there. The
balanced run later reached 0.272 without any of that.
