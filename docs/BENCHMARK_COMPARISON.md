# How our results compare to the OpenSDI paper

Compared against Wang et al., *OpenSDI: Spotting Diffusion-Generated Images in
the Open World*, CVPR 2025 ([arXiv:2503.19653](https://arxiv.org/abs/2503.19653)),
which is the paper that introduced the dataset we train on.

## What is not comparable, stated first

Four differences matter, and three of them favour the paper heavily:

| | paper | ours |
|---|---|---|
| task | **binary**, real vs fake | **3-class**, real / generated / edited |
| training images | **200,000** | 14,822 (**7.4%**) |
| encoder | fully trained | **frozen**, only a small decoder and head train |
| resolution | 224 detect, 512 localise | 448 both |

Our three-class task is strictly harder than their binary one, so for any
comparison against their numbers we collapse ours to real-vs-fake. We also report
balanced accuracy throughout, because our test split is 58% real while theirs is
50/50, and raw accuracy is not comparable across different class mixes.

## Detection

Balanced real-vs-fake accuracy, in-domain on SD1.5:

| method | accuracy | notes |
|---|---|---|
| MaskCLIP (paper's method) | **0.9272** | 200K training images, encoder trained |
| RINE | 0.9098 | |
| **ours, CLIP ViT-B/16 @448** | **0.8092** | 14.8K images, encoder frozen |
| IML-ViT | 0.7573 | |

We sit about 10 points behind the state of the art and above IML-ViT, which is a
published baseline in the same table. Given 7.4% of the training data and a
frozen backbone, that is a reasonable place to be rather than a disappointing one.

## Localisation

Mask IoU, in-domain on SD1.5:

| method | IoU |
|---|---|
| MaskCLIP | **0.6712** |
| IML-ViT | 0.6651 |
| TruFor | 0.6342 |
| **ours** | **0.2370** |

This is the honest weak point and the gap is large. They train the full network at
512px on 200,000 images; we train a four-block decoder on frozen features with
14,822. Localisation is the task that benefits most from both, so the ordering is
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

Ours, measured on the DINOv2 mask head:

| | sd15 | sd3 | sdxl | flux | loss |
|---|---|---|---|---|---|
| ai_edited recall | 0.670 | 0.583 | 0.550 | 0.505 | -25% |
| mask IoU | 0.353 | 0.268 | 0.155 | 0.111 | **-69%** |

The same asymmetry appears in both: roughly a quarter of classification lost
against roughly three quarters of localisation. A detector on an unfamiliar
generator still registers that something is wrong while largely losing track of
where. That is a property of the problem rather than of either implementation,
which is worth more than a headline number.

**Caveat:** our cross-generator figures were measured on the DINOv2 mask head, not
the stronger CLIP one, so they understate what the current configuration would do.
Re-running that evaluation on the CLIP model is outstanding.

## Our own progression

| configuration | 3-class balanced | binary | IoU |
|---|---|---|---|
| DINOv2 ViT-S linear probe | 0.7512 | - | - |
| DINOv2 ViT-S mask head @448 | 0.7292 | 0.7519 | 0.277 |
| CLIP ViT-B/16 mask head @224 | 0.7835 | 0.7878 | 0.176 |
| **CLIP ViT-B/16 mask head @448** | **0.7969** | **0.8092** | 0.237 |

Both jumps came from the representation and how it is fed. Six other levers were
tested and eliminated with measurements: encoder capacity (p = 1.000), epochs
(converges at 3), data volume, decision rule (+0.0002 across a full sweep),
resolution 448 to 672 (within noise), and class balance (-0.0036 prior-corrected).

Note that DINOv2 still holds the best IoU at 0.277 despite the worst
classification, because it ran on a 32x32 grid against CLIP's 28x28. Grid
resolution matters for localisation independently of feature quality.

## What the gap says to do next

**For detection**, the gap is mostly data and a frozen encoder. Scaling toward the
full 200,000 images is the obvious move and OpenSDI has them.

**For localisation**, the gap is larger and points at the same two causes plus
resolution. The paper localises at 512px with a fully trained network. Our 0.237
comes from a small decoder on frozen features, and an unfinished ViT-L/14 run at a
32x32 grid reached IoU 0.2545 after a single epoch, above the ViT-B/16 best of
0.237 across all ten of its epochs, so there is clearly headroom there.
