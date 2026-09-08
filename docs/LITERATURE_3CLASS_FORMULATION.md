# Is "real vs fully-AI-generated vs AI-edited" a formulation the literature supports?

Literature review, 2026-09-08. Every paper below was fetched and read (PDF or arXiv
abstract page), not taken from search snippets. Anything I could not confirm by fetching
is marked **UNVERIFIED**.

---

## Bottom line

**Yes — the 3-class formulation is real, current, and named in the literature. But every
paper that uses it pairs the "edited" class with a pixel-level localisation head, and
every paper that drops the localisation head reports near-total failure on that class.**

The two strongest single facts:

1. **SID-Set (SIDA, CVPR 2025)** and **So-Fake-Set (2025)** both define exactly
   `REAL / FULL_SYNTHETIC / TAMPERED`. **Gallina et al. (ACM MM 2026 DFF workshop)**
   builds precisely a "unified multiclass framework (real vs. fully generated vs.
   tampered)" plus a segmentation branch. So the formulation is not idiosyncratic.
2. On SID-Set, standard **image-level** synthetic-image detectors trained at 224×224 crop
   score **0.8 % – 6.9 % accuracy on the tampered class** while scoring 83–94 % on the
   fully-synthetic class. The model that gets 92.7 % on tampered (SIDA) runs at 1024×1024
   **with a mask-prediction branch**.

So: **keep the 3-class label space, but do not ship it as a flat 224×224 image-level
classifier.** Recommendation in §6.

---

## 1. Papers that DO the multi-class / unified formulation

### 1.1 HiFi-IFDL — the canonical hierarchy (verified in full)

- **Title:** *Hierarchical Fine-Grained Image Forgery Detection and Localization*
- **Authors:** Xiao Guo, Xiaohong Liu, Zhiyuan Ren, Steven Grosz, Iacopo Masi, Xiaoming Liu
- **Affiliations:** Michigan State University; Shanghai Jiao Tong; Sapienza Rome
- **Venue:** CVPR 2023, pp. 3155–3165
- **arXiv:** [2303.17111](https://arxiv.org/abs/2303.17111), submitted 30 Mar 2023
- **PDF fetched:** https://openaccess.thecvf.com/content/CVPR2023/papers/Guo_Hierarchical_Fine-Grained_Image_Forgery_Detection_and_Localization_CVPR_2023_paper.pdf
- **Code:** https://github.com/CHELSEA234/HiFi_IFDL (CVPR23 + IJCV24 extension)

**The taxonomy, exactly as printed in Fig. 2a / Fig. 6a:**

```
                        Forgery (root)
Level 1:   Fully-synthesized              Partial-manipulated
Level 2:   Diffusion   |  GAN             CNN-based  |  Image editing
Level 3:   Uncond. | Cond.  (per branch)
Level 4:   DDPM, DDIM, GDM, LDM, StarGANv2, HiSD, StyleGAN2-ada,
           StyleGAN3, STGAN, Faceshifter, splicing, inpainting, copy-move
```

Verbatim from the paper: *"At level 1, we separate forged images into fully-synthesized
and partial-manipulated."* Level 1 **is** the thesis's distinction. Classification is
hierarchical, not flat: *"the classification probability at a node of DDPM is conditioned
on the classification probability of all nodes in the path of Forgery→Fully
Synthesis→Diffusion→Unconditional→DDPM. This differs to prior work which assume a 'flat'
structure in which attributes are mutually exclusive."*

**Motivation, verbatim:** *"the computer vision community has made considerable efforts,
which however branch separately into two directions: detecting either CNN synthesis, or
conventional image editing. As a result, these methods may be ineffective when deploying
to real-life scenarios."* — i.e. HiFi-IFDL exists **because** the field is split. It is
evidence for both sides of the question.

**Crucially, it is not an image-level classifier.** HiFi-Net has three components:
multi-branch feature extractor, **localization module** (pixel mask, deep-metric-learning
objective), and detection module. Every forged image is paired with a **high-resolution
ground-truth forgery mask**. Reported metrics are AUC **and** F1 at *both* image level and
pixel level.

**Scale — this matters for a one-GPU project.** HiFi-IFDL dataset: 13 forgery methods,
**100,000 images each**; real images from FFHQ, AFHQ, CelebA-HQ, YouTube Faces, MS-COCO,
LSUN. Splits: **1,710K train / 15K val / 174K test**. Training: **400,000 iterations at
batch size 16** (8 real + 8 forged) ≈ 6.4 M image presentations.

**Its own stated limitation, verbatim:** *"the model that performs well on the conventional
image editing can generalize poorly on diffusion-based inpainting method."* And its listed
failure modes include: *"Inpainted images have small forgery regions."*

Selected numbers (Tab. 3, image-editing domain, fine-tuned localisation, AUC/F1 %):

| Method | Coverage | CASIA | NIST16 | Avg |
|---|---|---|---|---|
| SPAN | 93.7/55.8 | 83.8/40.8 | 96.1/58.2 | 91.2/51.6 |
| PSCC-Net | 94.1/72.3 | 87.5/55.4 | 99.6/81.9 | 93.7/69.8 |
| ObjectFormer | 95.7/75.8 | 88.2/57.9 | 99.6/82.4 | 94.5/72.0 |
| HiFi-Net | 96.1/80.1 | 88.5/61.6 | 98.9/85.0 | **94.6/75.5** |

### 1.2 SID-Set / SIDA — explicit 3-class, 100K per class (verified)

- **Title:** *SIDA: Social Media Image Deepfake Detection, Localization and Explanation
  with Large Multimodal Model*
- **Authors:** Zhenglin Huang, Jinwei Hu, Xiangtai Li, Yiwei He, Xingyu Zhao, Bei Peng,
  Baoyuan Wu, Xiaowei Huang, Guangliang Cheng
- **Venue:** CVPR 2025 (poster page: https://cvpr.thecvf.com/virtual/2025/poster/32427)
- **arXiv:** [2412.04292](https://arxiv.org/abs/2412.04292v3) (PDF fetched)
- **Code:** https://github.com/hzlsaber/SIDA

**Class structure, verbatim from the paper:** *"300K images (i.e., 100K real, 100K
synthetic, and 100K tampered images)"*. Source images: COCO, Flickr30k, MagicBrush.
Tampered images built by a 4-stage pipeline: GPT-4o extracts objects from captions →
Language-SAM produces object masks → replacement dictionary → Latent Diffusion inpaints.

**Metrics:** image-level accuracy and F1 for detection; **AUC, F1 and IoU for
localisation**. Input resized to **1024×1024**.

**Table 2 — per-class accuracy / F1 (%) on SID-Set. This is the single most decisive
table for the thesis question.** (Values outside parentheses = off-the-shelf; the
retrained baselines used *crop size 224*, batch 64, Adam.)

| Method | Year | Real Acc | Fully-synthetic Acc | **Tampered Acc** | Overall Acc |
|---|---|---|---|---|---|
| Gram-Net | 2020 | 70.1 | 93.5 | **0.8** | 54.8 |
| Fusing | 2022 | 85.1 | 34.0 | **2.7** | 40.6 |
| LNP | 2023 | 71.2 | 91.8 | **2.9** | 55.3 |
| LGrad | 2023 | 64.8 | 83.5 | **6.8** | 51.7 |
| CNNSpot | 2021 | 79.8 | 39.5 | **6.9** | 42.1 |
| FreDect | 2020 | 83.7 | 16.8 | **11.9** | 37.4 |
| AntifakePrompt | 2024 | 64.8 | 93.8 | **30.8** | 63.1 |
| UnivFD | 2023 | 68.0 | 62.1 | **64.0** | 64.7 |
| **SIDA-7B** | 2024 | 89.1 | 98.7 | **92.7** | 93.5 |
| **SIDA-13B** | 2024 | 89.6 | 98.5 | **92.9** | 93.6 |

Read that column. A conventional image-level synthetic-image detector is *not merely
weak* on the AI-edited class — it is at 1–7 %, i.e. it essentially never fires. Even after
fine-tuning on SID-Set (the parenthesised deltas, e.g. Gram-Net ↑89.1) the classifiers only
reach ~90 % on tampered while the mask-supervised model gets there natively.

**Table 3 — localisation on SID-Set (tampered class):**

| Method | AUC | F1 | IoU |
|---|---|---|---|
| MVSS-Net* | 48.9 | 31.6 | 23.7 |
| HiFi-Net* | 64.0 | 45.9 | 21.1 |
| PSCC-Net | 82.1 | 71.3 | 35.7 |
| LISA-7B-v1 | 78.4 | 69.1 | 32.5 |
| SIDA-7B | 87.3 | 73.9 | 43.8 |

Note HiFi-Net — the unified 3-class-hierarchy model of §1.1 — transfers to diffusion
inpainting at **IoU 21.1**. Consistent with its own stated limitation.

### 1.3 So-Fake-Set / So-Fake-R1 — explicit 3-way protocol at 2M scale (verified)

- **Title:** *So-Fake: Benchmarking and Explaining Social Media Image Forgery Detection*
- **Authors:** Zhenglin Huang, Xiangtai Li, Xi Yang, Bei Peng, Xiaowei Huang, Baoyuan Wu,
  Dacheng Tao, Ming-Hsuan Yang, Guangliang Cheng
- **arXiv:** [2505.18660](https://arxiv.org/abs/2505.18660), v1 24 May 2025, v5 31 Jul 2026
  (PDF fetched)
- **Code:** https://github.com/hzlsaber/So-Fake

Verbatim: *"Under a unified three-way protocol over REAL, FULL SYNTHETIC, and TAMPERED
images, So-Fake jointly evaluates authenticity detection, tampered-region localization,
and explanation."* And: *"In both datasets, images are labeled as REAL, FULL SYNTHETIC, or
TAMPERED."*

- **So-Fake-Set:** ~2M images, 35 generators, 12 social-media categories. Source images:
  COCO, Flickr30k, WIDER, Tumblr, OpenForensics.
- **So-Fake-OOD:** 100K out-of-domain benchmark from real platforms, commercial generators
  held out of training.
- Explicit motivation, verbatim: existing methods *"cannot jointly address locally tampered
  and full synthetic"* images.

### 1.4 Gallina et al. 2026 — literally the thesis's architecture (verified, 6 days old)

- **Title:** *From Detection to Localization: A Unified Forensics Framework for Fully
  Synthetic and Tampered Images*
- **Authors:** Annalisa Gallina, Marco Fiorucci, Marco Brigo, Federica Battisti,
  Lamberto Ballan (University of Padova)
- **Venue:** DFF Workshop, ACM Multimedia 2026
- **arXiv:** [2609.02640](https://arxiv.org/abs/2609.02640), submitted **2 Sep 2026** (PDF fetched)
- **Code:** https://github.com/anngal01/From-Detection-to-Localization-A-Unified-Forensics-Framework-for-Fully-Synthetic-and-Tampered-Images

Abstract, verbatim: *"Conventional approaches typically frame image manipulation detection
as a binary classification task (real vs. generated), which limits the capability to
distinguish and localize different forms of manipulation. To address these constraints,
this work extends an existing detector by introducing a unified multiclass framework (real
vs. fully generated vs. tampered). In addition to classifying image authenticity, the
framework incorporates a segmentation branch to enable pixel-level localization of
tampered regions."*

Architecture: **frozen DINOv2** backbone, features from multiple transformer blocks shared
between (a) a trainable importance estimator → 3-class head, and (b) a trainable decoder →
pixel mask. *"a three-class branch categorizes images as real, fully synthetic, or
partially tampered; the latter triggers a lightweight segmentation branch."* Trained 3
epochs (detection) + 5 epochs (segmentation), batch 128, lr 1e-3, Adam. **96.9 M
parameters, 370 MB, 16.4 ms/image on one NVIDIA L40s.** This is a one-GPU-sized project.

**Table 1 — So-Fake-Set:**

| Method | Year | Type | Det. Acc | Det. F1 | Loc. IoU | Loc. F1 |
|---|---|---|---|---|---|---|
| CnnSpot | 2021 | Detection | 89.6 | 87.7 | – | – |
| UnivFD | 2023 | Detection | 84.0 | 63.8 | – | – |
| NPR | 2024 | Detection | 81.8 | 61.5 | – | – |
| **HIFI-Net** | 2022 | IFDL | **39.0** | 25.2 | 12.1 | 18.3 |
| TruFor | 2023 | IFDL | 87.3 | 85.9 | 47.5 | 57.6 |
| PSCC-Net | 2022 | IFDL | 84.2 | 81.1 | 46.3 | 54.8 |
| SIDA | 2025 | LLM | 91.9 | 91.5 | 44.1 | 58.9 |
| So-Fake-R1 | 2025 | LLM | **93.2** | **92.9** | 48.6 | 63.9 |
| **Gallina et al.** | 2026 | IFDL | 92.4 | 92.4 | **77.8** | **83.9** |

**Table 2 — cross-dataset, trained on So-Fake, evaluated on SID-Set (no fine-tuning),
Acc/F1 %:** theirs 96.0/96.6 real, 99.7/99.7 fake, overall 97.7/97.7; cross-dataset
localisation IoU 72.95.

### 1.5 Model attribution / GAN fingerprinting (verified citation only)

- **Title:** *Attributing Fake Images to GANs: Learning and Analyzing GAN Fingerprints*
- **Authors:** Ning Yu, Larry Davis, Mario Fritz
- **Venue:** ICCV 2019
- **PDF:** https://openaccess.thecvf.com/content_ICCV_2019/papers/Yu_Attributing_Fake_Images_to_GANs_Learning_and_Analyzing_GAN_Fingerprints_ICCV_2019_paper.pdf
- **Code:** https://github.com/ningyu1991/GANFingerprints

Multi-class over *generators*, all fully-synthetic, all image-level. Relevant as prior art
for "the fake class can be subdivided", **not** for a real/generated/edited split. HiFi-IFDL
explicitly criticises this line as assuming a "flat" mutually-exclusive structure.

### 1.6 OpenForensics (verified citation only)

- **Title:** *OpenForensics: Large-Scale Challenging Dataset For Multi-Face Forgery
  Detection And Segmentation In-The-Wild*
- **Authors:** Trung-Nghia Le, Huy H. Nguyen, Junichi Yamagishi, Isao Echizen
- **Venue:** ICCV 2021 — **arXiv:** [2107.14480](https://arxiv.org/abs/2107.14480)
- 115K in-the-wild images, 334K faces, **face-wise segmentation masks, forgery boundaries,
  bounding boxes, landmarks**.

Note what it is: a *localised* GAN face-swap dataset whose unit of annotation is a mask,
not an image label. TruFor evaluates it as a **localisation** benchmark and reports that
*"most other methods fail catastrophically"* on it.

### 1.7 DeepFakeFace (verified citation only)

- *Robustness and Generalizability of Deepfake Detection: A Study with Diffusion Models*,
  Haixu Song, Shiyu Huang, Yinpeng Dong, Wei-Wei Tu, **arXiv:**
  [2309.02218](https://arxiv.org/abs/2309.02218) (2023).
- Face-only, image-level binary real/fake, no masks. **Does not support a 3-class
  formulation.** Listed here only because it was on the search list.

### 1.8 Other recent localized-AI-edit work (verified)

- **TGIF: Text-Guided Inpainting Forgery Dataset.** Hannes Mareen, Dimitrios Karageorgiou,
  Glenn Van Wallendael, Peter Lambert, Symeon Papadopoulos. **IEEE WIFS 2024.**
  **arXiv:** [2407.11566](https://arxiv.org/abs/2407.11566) v2, 4 Oct 2024 (PDF fetched).
  ~75K forged images (74,976) from 3,124 COCO originals, SD2 / SDXL / Adobe Firefly, up to
  1024×1024. Four sub-sets: **SD2-sp, PS-sp** (spliced) and **SD2-fr, SDXL-fr** (fully
  regenerated). See §4 — this is the sharpest evidence in the whole review.
- **AutoSplice: A Text-prompt Manipulated Image Dataset for Media Forensics.** Shan Jia et
  al., **CVPR 2023 Workshop on Media Forensics**, pp. 893–903. **PDF:**
  https://openaccess.thecvf.com/content/CVPR2023W/WMF/papers/Jia_AutoSplice_A_Text-Prompt_Manipulated_Image_Dataset_for_Media_Forensics_CVPRW_2023_paper.pdf
  3,621 manipulated + 2,273 authentic images, DALL·E 2 local inpainting from Visual News,
  **each with a manipulation mask**. (Details from the abstract page and repo, not the full
  PDF — mask provision is stated in both.)
- **DiffSeg30k: A Multi-Turn Diffusion Editing Benchmark for Localized AIGC Detection.**
  Hai Ci, Ziheng Peng, Pei Yang, Yingxin Xuan, Mike Zheng Shou. **arXiv:**
  [2511.19111](https://arxiv.org/abs/2511.19111), 24 Nov 2025. 30K diffusion-edited COCO
  images, 8 diffusion models, up to 3 sequential edits, **pixel-level annotations**.
  Verbatim from the abstract: *"Existing AIGC detection benchmarks focus on classifying
  entire images, overlooking the localization of diffusion-based edits."* They **reframe
  detection as semantic segmentation rather than binary classification.**
- **GIM: A Million-scale Benchmark for Generative Image Manipulation Detection and
  Localization.** Yirui Chen et al., **arXiv:** [2406.16531](https://arxiv.org/abs/2406.16531),
  24 Jun 2024 (rev. Jan 2025). >1M manipulated/real pairs built with SAM + LLM + generative
  models. Detection **and** localization.
- **SIDBench: A Python Framework for Reliably Assessing Synthetic Image Detection Methods.**
  Manos Schinas, Symeon Papadopoulos. **ACM MAD '24** workshop, 10 Jun 2024, Phuket.
  **arXiv:** [2404.18552](https://arxiv.org/abs/2404.18552). DOI 10.1145/3643491.3660277.
  Code: https://github.com/mever-team/sidbench. Note its own framing: SID is *"generating
  entirely synthetic images, which is a unique challenge compared to methods that alter
  portions of an image."* SIDBench is a **synthetic-image-detection-only** harness — it is
  the toolkit TGIF used to show SID methods cannot find local edits.

---

## 2. The separate-task view: different features, different metrics

### 2.1 ForensicHub — the field's own admission of "domain silos" (verified)

- **Title:** *ForensicHub: A Unified Benchmark & Codebase for All-Domain Fake Image
  Detection and Localization*
- **Authors:** Bo Du, Xuekang Zhu, Xiaochen Ma, Chenfan Qu, Kaiwen Feng, Zhe Yang,
  Chi-Man Pun, Jian Liu, Ji-Zhe Zhou
- **Venue:** **NeurIPS 2025, Datasets & Benchmarks track**
- **arXiv:** [2505.11003](https://arxiv.org/abs/2505.11003), 16 May 2025 (PDF fetched)

It splits the field into **four** domains, not two: Deepfake, **IMDL** (image manipulation
detection/localization), **AIGC** (fully AI-generated), and Document. Verbatim: *"the
research efforts of FIDL have gradually split into four relatively independent research
domains over time"*, causing *"significant domain silos, where each domain independently
constructs its datasets, models, and evaluation protocols without interoperability."*

Its Table 1 tabulates **Output Type** per method — this is the metric split, stated
directly:

| Task | Method | Backbone | Artifact strategy | **Output type** |
|---|---|---|---|---|
| Deepfake | Capsule-Net, RECCE, SPSL, UCF, SBI | VGG/Xception/EfficientNet | routing, reconstruction, phase spectrum, blending boundaries | **Label** |
| IMDL | MVSS-Net | ResNet | **BayarConv, Sobel** | Label, Mask |
| IMDL | CAT-Net | HRNet | **DCT** | **Mask** |
| IMDL | PSCC-Net | HRNet | multi-resolution conv | Label, Mask |
| IMDL | TruFor | SegFormer | **high-reso, multi-scale, edge** | Label, Mask |
| IMDL | IML-ViT | ViT | none | **Mask** |
| IMDL | Mesorch | ConvNeXt+SegFormer | **DCT** | **Mask** |
| AIGC | DIRE | ResNet | diffusion reconstruction | **Label** |
| AIGC | DualNet | CNN | **SRM, low frequency** | **Label** |
| AIGC | HiFiNet | HRNet | multi-branch | Label, Mask |
| AIGC | Synthbuster | – | Fourier | **Label** |
| AIGC | UnivFD | CLIP-ViT | none | **Label** |

Also verbatim on efficiency: *"Deepfake models are typically lightweight to support
real-time video detection, while IMDL models, which focus on pixel-level classification,
often adopt more complex and heavier architectures."*

**The IFF-Protocol result (Table 7) is the killer number for a 256×256 image-level
setup.** They train every model *jointly* on all four domains (FF++ / CASIAv2 / GenImage /
document sets, equal sampling, 20 epochs, **images resized to 256×256**, mask-outputting
models collapsed to a label by max-pooling) and report image-level AUC:

| Method | Columbia (IMDL) | IMD2020 (IMDL) | GenImage (AIGC) |
|---|---|---|---|
| ResNet | 0.482 | 0.533 | 0.797 |
| Xception | 0.465 | 0.537 | 0.980 |
| Swin | 0.636 | 0.631 | 0.999 |
| ConvNeXt | 0.625 | 0.598 | **1.000** |
| MVSS-Net | **0.298** | 0.539 | 0.994 |
| TruFor | **0.306** | 0.564 | 0.996 |
| IML-ViT | 0.483 | 0.556 | 0.991 |
| Mesorch | **0.285** | 0.570 | 0.996 |
| HiFiNet | 0.745 | 0.534 | 0.756 |
| FatFormer | **0.199** | 0.585 | 0.999 |
| Effort | 0.979 | 0.861 | 0.992 |
| FFDN | 0.553 | 0.624 | **1.000** |

At 256×256, image-level: **the fully-generated column is essentially solved (0.99–1.00);
the locally-manipulated columns sit at or below chance for most models.** Several
purpose-built IMDL models score *below 0.5* on Columbia once you take away their masks.

Their insight #4, verbatim: *"Less-explored backbones like ConvNeXt and Swin Transformer
outperform nearly all domain SoTAs under IFF-Protocol"* — i.e. the specialised forensic
machinery buys you nothing once you crush everything to 256×256 and one label.

### 2.2 SICA — "the Ji-Zhe phenomenon": artifacts do not transfer (verified)

- **Title:** *Can We Build a Monolithic Model for Fake Image Detection? SICA:
  Semantic-Induced Constrained Adaptation for Unified-Yet-Discriminative Artifact Feature
  Space Reconstruction*
- **Authors:** Bo Du, Xiaochen Ma, Xuekang Zhu, Zhe Yang, Chaoqun Niu, Chenfan Qu,
  Mingqi Fang, Zhenming Wang, Jingjing Liu, Jian Liu, Ji-Zhe Zhou
- **arXiv:** [2602.06676](https://arxiv.org/abs/2602.06676), v1 6 Feb 2026, v4 21 May 2026
  (PDF fetched)
- **Code:** https://github.com/venus-guangjian/SICA_OpenMMSec

Their Table 1, **"Overview of dominant artifacts across different domains"**, with an
explicit transferability column (✗ = conceptually non-transferable, △ = transferable in
principle but "barely work in other subdomains"):

| Domain | Dominant artifact | Dependency (constraint) | Transfer? | Rep. method |
|---|---|---|---|---|
| Deepfake | blending boundary | face-swapping mask | ✗ | Face X-ray |
| Deepfake | frequency | facial region | △ | F3-Net |
| Deepfake | physiology | human pulse (rPPG) | ✗ | FakeCatcher |
| **AIGC** | **global spectrum** | checkerboard pattern | ✗ | CNN-Gen |
| **AIGC** | **fingerprint** | GAN/gen. architecture | ✗ | Freq-Analysis |
| **AIGC** | **reconstruction** | diffusion prior | ✗ | DIRE |
| **IMDL** | **noise residual** | camera sensor (PRNU) | △ | ManTra-Net |
| **IMDL** | **edge** | boundary inconsistency | ✗ | MVSS-Net |
| Doc | text morphology | font/glyph rendering | ✗ | DocTamper |

Verbatim: *"though termed the same, artifacts in FID are highly distinct across
subdomains"* and *"A direct evidence for this collapse is that involving extra training
data from other subdomains will degrade model performance on the current subdomain."*

**Table 3(a), full fine-tuning, single-domain training, cross-domain AUC:**

| Train ↓ / Test → | Deepfake | AIGC | IMDL | Doc |
|---|---|---|---|---|
| Deepfake | 0.8183 | 0.7639 | **0.4605** | 0.3488 |
| **AIGC** | 0.5704 | **0.9291** | **0.4161** | 0.5263 |
| **IMDL** | 0.5862 | **0.4706** | **0.8535** | 0.7870 |
| Doc | 0.5604 | 0.4474 | 0.4843 | 0.8657 |
| **Unified** | 0.7900 | **0.8987** | 0.8223 | 0.8080 |

Two things to read here. (i) **AIGC→IMDL is 0.4161 and IMDL→AIGC is 0.4706 — both worse
than random.** A detector trained to spot fully-generated images is *anti*-correlated with
locally-manipulated images. (ii) **Unified training is not free**: AIGC drops 0.9291 →
0.8987 and Deepfake 0.8183 → 0.7900 relative to their own single-domain models. Verbatim:
*"training on one domain provides little benefit or conflict to another … artifacts cannot
be shared across domains."*

### 2.3 TruFor — the reference IML method (verified in full)

- **Title:** *TruFor: Leveraging All-Round Clues for Trustworthy Image Forgery Detection
  and Localization*
- **Authors:** Fabrizio Guillaro, Davide Cozzolino, Avneesh Sud, Nicholas Dufour,
  Luisa Verdoliva (Univ. Federico II Naples; Google Research)
- **Venue:** CVPR 2023, pp. 20606–20615
- **PDF fetched:** https://openaccess.thecvf.com/content/CVPR2023/papers/Guillaro_TruFor_Leveraging_All-Round_Clues_for_Trustworthy_Image_Forgery_Detection_and_CVPR_2023_paper.pdf
- **Project:** https://grip-unina.github.io/TruFor/

Features: **Noiseprint++**, a learned noise-sensitive camera-processing fingerprint trained
self-supervised **on real images only**, fused with RGB via a CMX/SegFormer transformer.
The paper's own comparison figure puts Noiseprint++ next to noiseprint and **SRM-filter
residuals**. The stated principle: *"Forgeries are detected as deviations from the expected
regular pattern that characterizes each pristine image."* Anomaly detection against a camera
model — **structurally incapable of flagging a fully-generated image, and structurally the
only thing that can flag a 2 %-of-pixels splice.**

**Metrics, verbatim:** *"As in most of the previous works, we measure pixel-level
performance in terms of F1 … for image-level analysis we use AUC … and balanced accuracy."*
Image-level score is *derived from* the localisation map: the detector module takes the
anomaly map and the confidence map as input.

**Table 1 — pixel-level F1 (best threshold / fixed 0.5):**

| Method | CASIAv1+ | Coverage | Columbia | NIST16 | DSO-1 | VIPP | OpenFor. | CocoGlide | **AVG** |
|---|---|---|---|---|---|---|---|---|---|
| ADQ (JPEG) | .494/.302 | .167/.165 | .401/.401 | .238/.146 | .483/.421 | .549/.457 | .644/.414 | .302/.300 | .410/.326 |
| Splicebuster (noise) | .252/.143 | .321/.192 | .811/.565 | .312/.174 | .662/.372 | .432/.260 | .459/.340 | .434/.332 | .460/.297 |
| ManTraNet | .320/.180 | .486/.317 | .650/.508 | .225/.172 | .537/.412 | .373/.255 | .661/.551 | .673/.516 | .491/.364 |
| SPAN | .169/.112 | .428/.235 | .873/.759 | .363/.228 | .390/.233 | .375/.223 | .176/.089 | .350/.298 | .391/.272 |
| CAT-Net v2 | .852/.752 | .582/.381 | .923/.859 | .417/.308 | .673/.584 | .672/.590 | .947/.899 | .603/.434 | .709/.601 |
| MVSS-Net | .650/.528 | .659/.514 | .781/.729 | .372/.320 | .459/.358 | .485/.389 | .225/.117 | .642/.486 | .534/.430 |
| PSCC-Net | .670/.520 | .615/.473 | .760/.604 | .210/.113 | .733/.458 | .309/.183 | .353/.105 | .685/.515 | .542/.371 |
| Noiseprint | .205/.137 | .342/.229 | .835/.513 | .345/.196 | .811/.439 | .546/.382 | .675/.420 | .405/.318 | .521/.329 |
| **TruFor** | .822/.737 | .735/.600 | .914/.859 | .470/.399 | .973/.930 | .746/.693 | .901/.827 | **.720/.523** | **.785/.696** |

**Table 2 — image-level AUC / balanced accuracy.** Note how much worse this is:
CocoGlide (diffusion inpainting on COCO) tops out at **TruFor 0.752 AUC / 0.639 balanced
accuracy**, and the paper says verbatim: *"many methods exhibit a very poor performance,
close to random guessing (0.5). This phenomenon is especially acute for accuracy."*
ManTraNet, SPAN, RRU-Net and AdaCFA all sit at exactly 0.500 balanced accuracy — they never
usefully decide at image level at all.

**CocoGlide** is TruFor's own contribution: **512 images generated from the COCO 2017
validation set using the GLIDE diffusion model**, at 256×256. Ships masks.

### 2.4 The rest of the IML canon (citations verified)

- **ManTra-Net: Manipulation Tracing Network for Detection and Localization of Image
  Forgeries With Anomalous Features.** Yue Wu, Wael AbdAlmageed, Premkumar Natarajan.
  **CVPR 2019, pp. 9543–9552.** PDF:
  https://openaccess.thecvf.com/content_CVPR_2019/papers/Wu_ManTra-Net_Manipulation_Tracing_Network_for_Detection_and_Localization_of_Image_CVPR_2019_paper.pdf
  Self-supervised pretext task = **classify 385 image manipulation types**; forgery
  localisation posed as **local anomaly detection** with a Z-score feature and an LSTM;
  **fully convolutional, handles images of arbitrary size** (no fixed 224 crop). Its
  artifact strategy is noise residual / PRNU-dependent (SICA Table 1).
- **SPAN** — TruFor's Table 3 and IML-ViT's Table 1 both catalogue it; verified only as
  a row in those tables, so treat the standalone citation as **UNVERIFIED** here.
- **MVSS-Net** — ICCV 2021; features **BayarConv + Sobel** (noise + edge), output
  `Label, Mask` (ForensicHub Table 1). Verbatim from HiFi-IFDL: *"The MVSS-Net uses
  multi-level supervision to balance between sensitivity and specificity."*
- **CAT-Net** — IJCV 2022; HRNet backbone; **DCT stream**, i.e. it learns **JPEG
  compression artifacts**; output `Mask` (ForensicHub Table 1). Best non-TruFor pixel F1
  in TruFor's Table 1 (avg .709/.601).
- **PSCC-Net** — TCSVT 2022; HRNet; multi-resolution progressive mask refinement; output
  `Label, Mask`.
- **IML-ViT** — see §4; output `Mask`, no hand-crafted artifact extractor at all.

Note the pattern across all of them: **SRM / BayarConv noise residuals, DCT / JPEG traces,
Sobel / boundary supervision, PRNU-style camera fingerprints.** None of these are the
features a synthetic-image detector uses (global spectrum, upsampling checkerboard,
diffusion reconstruction error, CLIP semantics). SICA Table 1 says so explicitly and marks
the transfer ✗.

### 2.5 A 2025 survey exists but I could not read it

*Unravelling Digital Forgeries: A Systematic Survey on Image Manipulation Detection and
Localization*, ACM Computing Surveys, DOI 10.1145/3731243. **ACM returned HTTP 403 — I
could not fetch it, so its content is UNVERIFIED.** ForensicHub (NeurIPS 2025 D&B) and
SICA (2026) cover the same ground and were read in full, so nothing rests on it.

---

## 3. What the standard forgery datasets expect as output

Sources: the dataset papers themselves where fetchable; IMD2020's own comparison table
(Table 1, which has a literal "Binary mask" column); the IMDLBenCo dataset index
(https://scu-zjz.github.io/IMDLBenCo-doc/imdl_data_model_hub/data/IMDLdatasets.html);
IML-ViT Table 2 for resolutions.

| Dataset | Auth / Tamp | Masks shipped? | Resolution (min–max) | Standard metric |
|---|---|---|---|---|
| **CASIA v1.0** | 800 / 921 | **No — not official.** Community 0-1 masks derived by subtracting original from tampered (Pham et al. 2019) | 256–384 | pixel F1 / AUC |
| **CASIA v2.0** | 7,491 / 5,123 | **No — not official.** Same; a corrected ground-truth repo exists (SunnyHaze/CASIA2.0-Corrected-Groundtruth) because the widely-circulated masks are noisy | 240–800 | pixel F1 / AUC |
| **Columbia** (uncompressed colour) | 183 / 180 | **Edge masks** marking spliced-object boundaries | 757×568 – 1152×768, TIFF/BMP | pixel F1 / AUC |
| **Coverage** | 100 / 100 | **Yes** — duplicated-region mask *and* forged-region mask, plus a tampering/similarity factor | 158–572 | pixel F1 / AUC |
| **NIST16 (Nimble 2016)** | 0 / 564 in the standard split | Yes | 480–**5616** | pixel F1 / AUC |
| **IMD2020** | 35,000 / 35,000 synthetic + 2,000 real-life | **Yes, official binary masks** for both subsets — they built the table to make this point | 176–**4437** | pixel F1 / AUC |
| **DEFACTO** | built from MS-COCO | **Yes, multiple masks per image** (see below) | 120–640 | pixel F1 / AUC |
| **tampCOCO** | 0 / 800,000 | Yes, official 0-1 masks | 51–640 | pixel F1 / AUC |
| **CocoGlide** | 512 tampered | Yes | 256×256 | pixel F1 (TruFor Table 1) |
| **AutoSplice** | 2,273 / 3,621 | Yes, one mask per generated image | — | pixel-level + image-level |

**Every single one of them is a localisation benchmark.** Not one of them is scored
primarily by image-level accuracy in the literature. TruFor's Table 1 (pixel F1) is the
headline table; Table 2 (image-level AUC/accuracy) is the secondary one, and TruFor is
unusual in reporting it at all.

### 3.1 DEFACTO — verified from the EUSIPCO PDF

- **Title:** *DEFACTO: Image and Face Manipulation Dataset*
- **Authors:** Gaël Mahfoudi (ICD, Univ. of Technology of Troyes), Badr Tajini (EURECOM),
  Florent Retraint (ICD/UTT), Frédéric Morain-Nicolier (CReSTIC, Univ. Reims
  Champagne-Ardenne), Jean-Luc Dugelay (EURECOM), Marc Pic (SURYS)
- **Venue:** **2019 27th European Signal Processing Conference (EUSIPCO)**, ISBN
  978-9-0827-9703-9
- **PDF fetched:** https://www.eurasip.org/Proceedings/Eusipco/eusipco2019/Proceedings/papers/1570533790.pdf
- **Site:** https://defactodataset.github.io
- Funded by ANR project DEFACTO ANR-16-DEFA-0002 and the French DGA.

**Built on COCO — verbatim:** *"The dataset was automatically generated using Microsoft
common object in context database (MSCOCO) to produce semantically meaningful forgeries."*
And: *"To produce meaningful forgeries, we took advantages of MSCOCO dataset. … Those
annotations include the segmentation of the objects that we use as a base to produce our
forgeries. The raw segmentation annotation cannot be used directly … We employed an alpha
matting technique to refine the masks."*

**Table I — images per category:** copy-move 19,000 · **inpainting 25,000** · splicing
105,000 · morphing 80,000. Total >200,000.

**Annotations, verbatim (§III.B):**
1. *"General information: for each image, a detailed JSON file is provided. In this file,
   every operation made on the ground truth images are listed. Parameters used by each
   operation are detailed."*
2. *"Localization: every image is also accompanied by one or more ground truth binary
   masks. One binary mask serves to localize the forgery under the probe mask directory.
   For splicing, copy-move, face morphing and swapping, a binary mask under the donor mask
   directory gives the localization of the source. Object-removal has a binary mask under
   the inpaint mask directory which localize what has been filled by the inpainting
   algorithm."*

Also worth quoting for the thesis: *"As the methods to detect those categories can be quite
different, we decided to first construct a dataset where each image as only been forged
using one of those categories only."* The dataset authors themselves assumed per-category
detectors.

### 3.2 IMD2020 — verified from the WACVW PDF

- **Title:** *IMD2020: A Large-Scale Annotated Dataset Tailored for Detecting Manipulated
  Images*
- **Authors:** Adam Novozámský, Babak Mahdian, Stanislav Saic (Institute of Information
  Theory and Automation, Czech Academy of Sciences)
- **Venue:** **IEEE/CVF WACV 2020 Workshops**, pp. 71–80, March 2020
- **PDF fetched:** https://openaccess.thecvf.com/content_WACVW_2020/papers/w4/Novozamsky_IMD2020_A_Large-Scale_Annotated_Dataset_Tailored_for_Detecting_Manipulated_Images_WACVW_2020_paper.pdf
- **Site:** http://staff.utia.cas.cz/novozada/db
- Extended version: *Extended IMD2020*, IET Biometrics, 2021, DOI 10.1049/bme2.12025
  (citation from search; **contents UNVERIFIED**)

Verbatim from the abstract: 2,322 camera models → 35,000 real images; the same number
manipulated *"by using a large variety of core image manipulation methods as well as
advanced ones such as GAN or Inpainting resulting in a dataset of 70,000 images"*; plus
*"2,000 'real-life' (uncontrolled) manipulated images … made by unknown people and
downloaded from Internet"* with originals located. And: *"We also manually created binary
masks localizing the exact manipulated areas of these images."*

Their Table 1 is literally a survey of datasets with a **"Binary mask"** yes/no column —
CoMoFoD Yes, MICC No, Columbia No, CASIA No, CASIA v2.0 No, REWIND Yes, Nimble 2017 Yes,
Realistic Tampering Yes, Coverage Yes, **IMD2020 both subsets Yes**. The whole point of the
contribution is the masks.

### 3.3 The COCO overlap — state it plainly

**DEFACTO, CASIA, tampCOCO, CocoGlide, TGIF, SID-Set, So-Fake-Set, DiffSeg30k and GIM are
all wholly or partly built on MS-COCO.** DEFACTO's forgeries *are* COCO instance
segmentations refined by alpha matting; CocoGlide is COCO 2017-val inpainted with GLIDE;
TGIF is 3,124 COCO originals; SID-Set's tampered split is COCO + Flickr30k + MagicBrush.
Two consequences for a thesis:

- Any "real" class drawn from COCO and any "edited" class built by inpainting COCO share
  the *same* underlying photographs. A model can only separate them on manipulation traces,
  never on content — which is the honest version of the task, but also the reason
  image-level accuracy collapses when you resize away the traces.
- Conversely, if the "fully generated" class is *not* COCO-derived (most AIGC sets are
  LSUN/FFHQ/LAION), a 3-class classifier can trivially separate class 2 on **semantics and
  content distribution**, not on generation artifacts, and the accuracy will look excellent
  and mean nothing. Guard against this explicitly.

---

## 4. Is image-level classification of "edited" even well-posed?

This is where the evidence is most one-sided.

### 4.1 The direct experiment: TGIF (WIFS 2024)

TGIF benchmarks 8 IFL (localisation) methods and 12 SID (synthetic-detection) methods on
the same images. Two verbatim findings:

> *"We do not include the performance results for the spliced datasets (SD2-sp and PS-sp),
> since we cannot expect SID to detect these types of local manipulation, as they were not
> designed for this purpose. **We have verified that, indeed, no SID method is able to
> detect the synthetic spliced regions.**"*

> *"our benchmark analysis shows that some of the existing IFL methods are able to detect
> and localize spliced images, whereas they fail to localize the inpainted area in fully
> regenerated images. In contrast, some of the existing SID methods are able to detect
> fully regenerated images, yet lack the ability to localize the synthetic inpainted area.
> These limitations highlight the need for new forensic methods, leveraging elements from
> both IFL and SID methods."*

**Table I — IFL pixel F1:**

| Method | SD2-sp | PS-sp | SD2-fr | SDXL-fr |
|---|---|---|---|---|
| PSCC-Net | 0.15 | 0.38 | 0.05 | 0.05 |
| SPAN | 0.00 | 0.00 | 0.00 | 0.00 |
| ImageForensicsOSN | 0.23 | 0.36 | 0.20 | 0.18 |
| MVSS-Net++ | 0.07 | 0.08 | 0.06 | 0.09 |
| ManTra-Net | 0.15 | 0.56 | 0.03 | 0.05 |
| **CAT-Net** | **0.87** | **0.85** | 0.04 | 0.03 |
| **TruFor** | **0.83** | **0.79** | 0.19 | 0.18 |
| MMFusion | 0.75 | 0.74 | 0.15 | 0.18 |

**Table II — SID image-level AUC** (spliced columns omitted by the authors *because no SID
method works there at all*): DIMD 1.00 (SD2-fr) / 0.94 (SDXL-fr); PatchCraft 0.98 / 0.95;
RINE 0.89 / 0.89; UnivFD 0.82 / 0.80; LGrad 0.85 / 0.83; DIRE 0.49 / 0.62; CNNDetect
0.57 / 0.61.

Why the fr column kills IFL, verbatim: *"regenerating the image removes most invisible
traces such as compression artifacts and camera noise. In fact, diffusing and regenerating
an image is an existing attack against forgery detection methods."*

**Compression destroys the IFL side entirely.** CAT-Net on SD2-sp: **0.87 uncompressed →
0.05 at JPEG Q80**. TruFor: 0.83 → 0.32. On the SID side only two of twelve methods survive
JPEG and only one survives WEBP.

### 4.2 The direct experiment on resolution: IML-ViT

- **Title:** *IML-ViT: Benchmarking Image Manipulation Localization by Vision Transformer*
- **Authors:** Xiaochen Ma, Bo Du, Zhuohang Jiang, Xia Du, Ahmed Y. Al Hammadi, Jizhe Zhou
- **arXiv:** [2307.14863](https://arxiv.org/abs/2307.14863), 27 Jul 2023, rev. 24 Nov 2024
  (PDF fetched)
- **Code:** https://github.com/SunnyHaze/IML-ViT

The paper's design thesis, verbatim: *"artifacts are sensitive to image resolution,
amplified under multi-scale features, and massive at the manipulation border."* And on
resizing specifically:

> *"While semantic segmentation and IML share similar inputs and outputs, IML tasks are
> more information-intensive, focusing on detailed artifacts rather than macro-semantics at
> the object level. Existing methods use various extractors to trace artifacts, but **their
> resizing methods already harm these first-hand artifacts**. Therefore, preserving the
> original resolution of the images is crucial to retain essential artifacts for the model
> to learn."*

Implementation: **pad every image to 1024×1024** (only images exceeding that get resized,
longest side to 1024, aspect preserved).

**Table 9 ablation, trained 200 epochs on CASIAv2. "w/o high resolution" = resize
everything to 512×512 instead of the 1024 padding.** Extracted with layout preserved:

| Setting | Init | CASIAv1 F1/AUC | Coverage F1/AUC | Columbia F1/AUC | NIST16 F1/AUC | **Mean F1/AUC** |
|---|---|---|---|---|---|---|
| w/o MAE | Xavier | .1035/– | .0439/– | .0744/– | .0632/– | .0713/– |
| w/o MAE | ViT-B ImNet-21k | .5820/.9037 | .2123/.7898 | .5040/.8335 | .2453/.7939 | .3859/.8302 |
| **w/o high resolution** | MAE ImNet-1k | .5747/.9121 | .2622/.7889 | .5150/.8028 | .3292/.7950 | **.4153/.8247** |
| w/o multi-scale | MAE ImNet-1k | .6504/.9306 | .3877/.8829 | .7096/.8816 | .2847/.7771 | .5081/.8681 |
| w/o edge-supervision | MAE ImNet-1k | .6177/.9240 | .3176/.8789 | .6843/.9161 | .2648/.8045 | .4711/.8809 |
| **Full setup** | MAE ImNet-1k | .7206/.9420 | .4099/.9137 | .7798/.9337 | .3317/.8064 | **.5605/.8990** |

**Going from 1024-pad to 512-resize costs 14.5 points of mean pixel F1 (0.5605 → 0.4153) —
the single largest ablation in the table, larger than removing edge supervision or the
feature pyramid. And that is 512, not 224.**

**Their Table 2 also gives the manipulated-area statistics that explain why:** *"On average,
CASIAv2 has 7.6 % of pixels as tampered areas, while Defacto has only 1.7 %."* At 224×224
a 1.7 % region is ~840 pixels; after a ViT patch embed at stride 16 that is roughly **3
patches**. There is no signal budget left.

Dataset resolutions from the same table: CASIAv2 240–800, CASIAv1 256–384, **NIST16
480–5616**, Coverage 158–572, DEFACTO 120–640, Columbia 568–1152, **IMD-20 176–4437**,
JPEG RAISE 1515–6159. IML datasets are high-resolution on purpose.

### 4.3 The direct experiment on 224 for the *3-class* task: Gallina et al. 2026

Their ablation is the closest thing in the literature to running the thesis's exact
experiment twice. Same 3-class head, same data (SIDA-Set train split, evaluated on its test
split); only the backbone and preprocessing differ. Verbatim: *"for CLIP, images are first
center-cropped to 512 × 512 pixels and then resized to 224 × 224 pixels. These distinct
strategies are motivated by the characteristics of the training dataset: **aggressively
cropping high-resolution images to 224 × 224 pixels can discard relevant fine-grained
details, potentially affecting both detection and localization performance**."*

**Table 3 — per-class accuracy (%):**

| Backbone | Real | Fake (fully synthetic) | **Tampered** | Overall |
|---|---|---|---|---|
| CLIP ViT-L/14 (crop 512 → **resize 224**) | 95.1 | 95.9 | **69.0** | 86.6 |
| DINOv2 (crop **518**, no downscale) | 93.1 | 99.8 | **93.3** | 95.4 |

**The tampered class moves 69.0 → 93.3 (+24.3 points) while real barely moves (−2.0) and
fake moves +3.9.** Honest caveat: backbone and preprocessing change together, so this is
not a clean resolution-only ablation — but the direction, the size, and the fact that the
gain is concentrated almost entirely in the *tampered* class are exactly what the
resolution hypothesis predicts, and the authors attribute it to the cropping.

Their failure analysis, verbatim: *"The main issue stems from the aggressive cropping needed
to meet the input constraints of DINOv2: when most of the manipulated region falls outside
the cropped area, the model may have difficulty detecting the remaining visible portion.
Additionally … **when the tampered region is very small, the model tends to highlight a
slightly larger surrounding area**."*

### 4.4 Converging evidence

- **SIDA Table 2** (§1.2): SID detectors retrained at **crop size 224** get 0.8–11.9 %
  accuracy on the tampered class.
- **ForensicHub Table 7** (§2.1): everything resized to **256×256**, image-level only →
  IMDL columns at or below chance while GenImage hits 1.000.
- **HiFi-IFDL** (§1.1) lists *"Inpainted images have small forgery regions"* as one of its
  three named failure modes, and admits poor generalisation to diffusion inpainting.
- **TruFor** (§2.3) cites prior work that analyses *"the whole image avoiding resizing (so
  as not to lose precious forensics traces) through a gradient checkpointing technique"* —
  the field pays real engineering cost specifically to avoid downsampling.
- **DiffSeg30k** (§1.8) reframes the whole problem as segmentation because
  *"existing AIGC detection benchmarks focus on classifying entire images, overlooking the
  localization of diffusion-based edits."*

**Verdict on well-posedness.** Image-level "edited vs real" is well-posed *as a label*, and
the literature does define it. It is **not learnable from a 224×224 resize** for anything
but large or crude edits. Nobody in the 2023–2026 literature achieves it that way; every
method that achieves it either keeps ≥512 px (Gallina 518, SIDA 1024, IML-ViT 1024,
TruFor full-resolution) or attaches a pixel-level mask loss, and usually both.

---

## 5. Where each side of the question actually lands

**Does the field support a unified 3-class formulation?** Yes, and increasingly so:
HiFi-IFDL (CVPR'23) level 1, SID-Set (CVPR'25), So-Fake (2025), Gallina et al. (ACM MM'26).
The exact label triple `REAL / FULL_SYNTHETIC / TAMPERED` is written down in at least two
benchmark papers.

**Does the field also treat (a) and (b) as fundamentally different tasks?** Also yes, and
the papers that unify them say so loudest:

- HiFi-IFDL's opening sentence: efforts *"branch separately into two directions."*
- ForensicHub (NeurIPS'25 D&B): *"domain silos … each domain independently constructs its
  datasets, models, and evaluation protocols without interoperability."*
- SICA (2026): the Ji-Zhe phenomenon; AIGC↔IMDL transfer AUC of 0.42/0.47, i.e. worse than
  chance; unified training *degrades* single-domain performance.
- TGIF (WIFS'24): SID methods provably cannot see local edits; IFL methods provably cannot
  see fully-regenerated ones.

The reconciliation is precise and it is the whole answer to the thesis question:

> The **label space** unifies cleanly. The **feature space** does not. Papers that unify the
> label space and keep two output heads (one image-level, one pixel-level) succeed. Papers
> that unify the label space and keep only an image-level head fail on the edited class.

---

## 6. Recommendation for a one-GPU thesis

**Keep the 3-class label space. Add a segmentation head. Do not split into two models.**

Reasons, each tied to evidence above:

1. **Splitting into two models is the option with the weakest support.** Nobody argues for
   two deployed models; they argue that the two *feature families* differ. ForensicHub's
   own motivation: *"in real-world scenarios, it is often impossible to predetermine the
   type of manipulation present in an image, making unified detection particularly
   important for users."* And SICA shows ensembling suffers the *"barrel effect"* from
   *"error propagation and the routing bottleneck."*
2. **A flat 3-class image-level classifier at 224×224 will produce a near-zero recall
   "edited" class, and you can predict the number before you run it** — 0.8 % to 6.9 %
   (SIDA Table 2), or ≤0.5 AUC (ForensicHub Table 7). If the current project already
   reports a healthy edited-class accuracy at 224, that is a red flag to check for the COCO
   / non-COCO content shortcut described in §3.3, not a result.
3. **The mask loss is what makes the third class learnable, and it is nearly free.**
   Gallina et al. get IoU 77.8 (vs. 48.6 for the best prior method) with a **frozen DINOv2
   and 96.9 M trainable-plus-frozen parameters, 370 MB, 16.4 ms/image on one L40s**, trained
   for 3 + 5 epochs. That is a single-GPU budget. It is also, structurally, exactly the
   3-class-plus-segmentation-branch design.
4. **Do not attempt HiFi-IFDL's full hierarchy.** 1.71 M training images and 400K
   iterations at batch 16 is out of scope, and HiFi-Net's own transfer numbers (IoU 21.1 on
   SID-Set, 39.0 detection accuracy / 12.1 IoU on So-Fake-Set) show the hierarchy does not
   buy generalisation to diffusion inpainting.

Concretely:

- **Architecture:** frozen self-supervised backbone (DINOv2 ViT-L, or ViT-B if memory is
  tight) → shared multi-block features → (a) 3-way head, (b) light upsampling decoder to a
  binary tamper mask. Mask loss = BCE + Dice (Dice matters: manipulated pixels are a small
  minority; Gallina et al. and SIDA both use this combination).
- **Resolution:** at minimum 512; 518 if you use DINOv2's patch-14 geometry. **Never 224.**
  If VRAM forces a compromise, prefer *cropping* to 512 over *resizing* to 224 — but note
  Gallina et al.'s failure mode where the edit falls outside the crop, so tile-and-vote at
  test time.
- **Metrics — report both levels, and never report only overall accuracy.** Image level:
  per-class accuracy and F1 for all three classes plus the macro average (this is exactly
  how SIDA Table 2 and Gallina Table 3 are laid out, and it is what exposes a dead class).
  Pixel level, on the tampered class only: F1, AUC, IoU (SIDA Table 3; TruFor Tables 1–2;
  IML-ViT).
- **Data:** SID-Set (300K, 100K/class, already exactly your label space) is the natural
  training set; So-Fake-OOD (100K, held-out commercial generators) or TGIF is the natural
  out-of-domain test. For a classical-forgery sanity check use CASIAv1 + Columbia +
  Coverage + IMD2020 with pixel F1; for diffusion inpainting use CocoGlide, AutoSplice,
  TGIF and DiffSeg30k. Keep the edited-region area distribution in your results table — a
  method that only works above ~7 % tampered area (CASIAv2-like) and dies at 1.7 %
  (DEFACTO-like) should be reported as such.
- **Add compression to the evaluation.** TGIF: CAT-Net 0.87 → 0.05 at JPEG Q80. Any result
  on uncompressed images overstates real-world performance by an order of magnitude.

**The one reframe worth considering** if segmentation supervision turns out to be too
expensive: follow DiffSeg30k and drop the image-level formulation entirely — treat the whole
problem as 3-way *semantic segmentation* (real pixels / generated-everywhere / edited
region), where the "fully generated" class is just a mask covering the entire image. It
gives you the same three answers, trains with one loss, and is exactly the direction the
2025–2026 literature is moving.
