# What actually generalises in AI-image detection — a literature review

Compiled 2026-09-08. Every citation below was checked by fetching the arXiv abstract page,
the publisher page or the paper HTML. Where a number could not be confirmed from a primary
source it is marked **[unverified]**.

Context: this project is a 3-class classifier (real / ai_generated / ai_edited) over ~78k
images from 20 corpora where **corpus identity perfectly predicts the label**, ConvNeXt-Small
fine-tuned end-to-end at 224x224, 89% on its own split, failing on arbitrary photographs.
Two shortcut signatures are already verified locally: augmentation strength is inversely
monotone with validation accuracy, and source native resolution separates corpora by ~4σ in
post-resize high-frequency energy.

---

## 1. The generalisation gap, quantified

### 1.1 The canonical in-domain vs cross-generator numbers

**GenImage** — Zhu et al., *GenImage: A Million-Scale Benchmark for Detecting AI-Generated
Image*, NeurIPS 2023 Datasets & Benchmarks, arXiv:2306.08571
(https://arxiv.org/abs/2306.08571). Eight generators (BigGAN, GLIDE, VQDM, SD v1.4, SD v1.5,
ADM, Midjourney, Wukong). Trained and tested within one generator, ResNet-50 gets
**98.5–99.9%**. Trained on one generator and tested on the others, the average falls to
roughly **60–70%**, and for dissimilar generator pairs to near chance (~50%).

That single table is the shape of the whole field: ~99% in-domain, ~65% cross-generator.

### 1.2 What happens when you leave the lab entirely

**Deepfake-Eval-2024** — Chandra, Lee, Murtfeldt, Qiu, Karmakar, Tanumihardja, Farhat,
Caffee, Lee, Choi, Paik, Kim, Etzioni, arXiv:2503.02857 (https://arxiv.org/abs/2503.02857).
In-the-wild deepfakes collected in 2024 from 88 websites in 52 languages. For the **image**
detectors, measured on their original academic benchmarks vs on this set:

| Detector | Original benchmark AUC | Deepfake-Eval-2024 AUC |
|---|---|---|
| UFD (UnivFD, CLIP backbone) | 0.94 | **0.56** |
| DistilDIRE | 0.99 | **0.52** |
| NPR | 0.98 | **0.53** |

Headline: AUC drops ~45% for image models. Fine-tuning on 60% of the in-the-wild set and
testing on the other 40% recovered only about **+4.5% AUC on average** — the domain is hard,
not merely unseen.

### 1.3 Independent, zero-shot, large-scale

**Ren, Zhou, Shen, Zewde, Duong et al.**, *How well are open sourced AI-generated image
detection models out-of-the-box: A comprehensive benchmark study*, arXiv:2602.07814,
February 2026 (https://arxiv.org/html/2602.07814v1). 16 detection methods / 23 pretrained
variants, 12 datasets, 2.6M images, 291 generators, **no fine-tuning**:

- Best: **Community-Forensics 78.0% mean accuracy** (82.1% median)
- SAFE 68.8%, PatchCraft 67.5%
- UnivFD **40.7%**, CNNSpot **37.5%** — i.e. worse than chance-on-balanced-data in aggregate
- 40.5 percentage points between best and worst
- On modern commercial generators, most detectors get **18–30%** detection accuracy
- Ranking is unstable across datasets (Spearman ρ between dataset rankings 0.01–0.87)
- "Training data alignment critically impacts generalization, causing 20–60% performance
  variance within identical architectures"

### 1.4 The collapse is asymmetric — real accuracy holds, fake accuracy dies

**Li, Yan, He, Zeng, Jiang, Xiong, Fu**, *Is Artificial Intelligence Generated Image
Detection a Solved Problem?* (AIGIBench), NeurIPS 2025 Datasets & Benchmarks,
arXiv:2505.12335 (https://arxiv.org/abs/2505.12335). Eleven detectors (ResNet-50,
CNNDetection, Gram-Net, LGrad, CLIPDetection, FreqNet, NPR, DFFreq, LaDeDa, AIDE, SAFE):

- SAFE: **100% real-accuracy / 99.9% fake-accuracy on ProGAN in-domain** → **16.4%
  fake-accuracy on SocialRF** (real social-media images)
- ResNet-50: 98.1% → **13.4%** fake-accuracy on SocialRF
- On face-swap data, detectors "misclassify nearly all samples as real"
- Common augmentations (rotation, colour jitter, masking) give **limited benefit and
  sometimes trade off** real vs fake accuracy
- Cropping beats resizing at test time, but the gain lands almost entirely on real-accuracy

This is exactly the failure mode described for this project: an arbitrary user photograph is
out of every training corpus, and the model has no "looks like COCO" evidence to call it real.

### 1.5 Per-generator spread

**Bernabeu-Pérez, Lopez-Cuena, Garcia-Gasulla**, *Present and Future Generalization of
Synthetic Image Detectors*, arXiv:2409.14128 (https://arxiv.org/abs/2409.14128):

- A detector trained on SD1.x: **98.30% recall on its own generator, 27.59% on DALL·E 3**
- Recall on unseen generators ranges from 94% down to 24%
- **45.22% average recall on closed generators** (DALL·E, Midjourney, Firefly) vs
  **76.60% on open ones**
- On their in-the-wild set, **no tested detector exceeds 50% recall on both authentic and
  synthetic images simultaneously**
- Their first guideline: "generalization should never be assumed in the field of SID"

### 1.6 The best honest numbers anyone reports

- **LaDeDa** (Cavia, Horwitz, Reiss, Hoshen, arXiv:2406.09398,
  https://arxiv.org/abs/2406.09398): ~99% mAP in-domain → **93.7% mAP on WildRF**, their own
  social-media in-the-wild set. The authors still conclude real-world detection "is still
  unsolved".
- **Community Forensics** (Park & Owens, CVPR 2025, arXiv:2411.04125,
  https://arxiv.org/abs/2411.04125): trained on 2.7M images from **4,803 generators**;
  self-reports 0.987 mAP / **89.2% accuracy on 21 unseen models**, and 0.904 mAP / **81.8%
  accuracy on Synthbuster**. Note the tension with §1.3, where the same method measured
  independently on 12 datasets gets 78.0% mean accuracy.
- **B-Free** (see §4.2): 96.4% average balanced accuracy across 27 generators — but on
  curated generator outputs, not laundered social-media images.

### 1.7 What to tell the student the target is

A defensible, honest set of targets for a well-executed detector in 2026:

| Setting | Realistic number |
|---|---|
| In-domain, same generators, same corpora | 95–99% (and meaningless) |
| Held-out **generator**, same clean pipeline | **80–90% accuracy / 0.90–0.97 AUC** |
| Held-out **source corpus** (this project's real problem) | **65–80% accuracy** |
| Truly in-the-wild, recompressed, social-media laundered | **0.55–0.70 AUC** — i.e. barely better than a coin |

**89% on a random split of 20 single-class corpora is not on this table at all.** It is not a
generalisation number. Under a leave-one-source-out protocol this project should *expect*
something in the 60s or low 70s, and that would be a perfectly respectable, publishable
result — because it would be the first number in the repo that measures the thing the
project claims to measure.

---

## 2. What actually transfers

### 2.1 CNNDetection — the foundational result and its expiry date

Wang, Wang, Zhang, Owens, Efros, *CNN-generated images are surprisingly easy to spot... for
now*, CVPR 2020, arXiv:1912.11035 (https://arxiv.org/abs/1912.11035). Train a ResNet-50 on
ProGAN only, with aggressive JPEG and blur augmentation, and it transfers across 11 CNN
generators. Mechanism: a shared up-sampling/checkerboard fingerprint in the CNN decoder.

Cost: full backbone training. **Status in 2026: obsolete.** It gets 37.5% mean accuracy
zero-shot in the independent 2026 benchmark (§1.3), and diffusion models broke it
specifically — Ricker, Damm, Holz, Fischer, *Towards the Detection of Diffusion Model
Deepfakes*, VISAPP 2024, arXiv:2210.14571 (https://arxiv.org/abs/2210.14571) show
state-of-the-art GAN detectors cannot reliably separate real from diffusion images, but
become near-perfect once retrained on diffusion data. The fingerprint is real; it is just
generator-family-specific.

### 2.2 UnivFD — frozen CLIP + linear probe

Ojha, Li, Lee, *Towards Universal Fake Image Detectors that Generalize Across Generative
Models*, CVPR 2023, arXiv:2302.10174 (https://arxiv.org/abs/2302.10174).

Mechanism: do **not** train a feature space on real-vs-fake at all. Take frozen CLIP ViT-L/14
image features (768-d) and fit either a nearest-neighbour classifier against real/fake
feature banks, or a **single linear layer**. Their argument: a trained detector's feature
space is organised around the artefacts of the training generator, so unseen generators fall
outside it; a general-purpose pretrained space is not, so the decision boundary transfers.
They state directly that "the higher the capacity, the easier it is for that model to overfit"
to training artefacts.

Numbers from the paper: on GAN test sets, linear probe ~99.31 mAP vs Wang et al. ~94.19 mAP;
on unseen diffusion/autoregressive models, linear probe ~95.00 mAP vs Wang et al. ~75.51 mAP.
Abstract claims **+15.07 mAP and +25.90% accuracy** on unseen diffusion/AR models.

Cost: **only the linear layer is trained.** The official repo
(https://github.com/WisconsinAIVision/UniversalFakeDetect) confirms this — training uses a
`--fix_backbone` flag so "only the linear layer's parameters will be trained."

Caveat, and it matters: this same method scores **40.7% mean accuracy** in the independent
2026 zero-shot benchmark (§1.3), **52.4% balanced accuracy on Synthbuster** (§4.2), and
**0.56 AUC on Deepfake-Eval-2024** (§1.2). UnivFD as published in 2023 does not survive to
2026. The *principle* survives; the *checkpoint* does not.

### 2.3 Frequency / spectral

- **Synthbuster** — Bammey, *Synthbuster: Towards Detection of Diffusion Model Generated
  Images*, IEEE Open Journal of Signal Processing, 2024
  (https://signalprocessingsociety.org/publications-resources/ieee-open-journal-signal-processing/synthbuster-towards-detection-diffusion,
  code https://github.com/qbammey/synthbuster). Fourier-domain artefacts of the diffusion
  process. Its companion dataset — 9 generators × 1000 images, loosely based on RAISE-1K —
  is a corpus this project already uses.
- **LGrad** — Tan, Zhou, Wei, Guo, Liu, Zhao, *Learning on Gradients: Generalized Artifacts
  Representation for GAN-Generated Images Detection*, CVPR 2023
  (https://openaccess.thecvf.com/content/CVPR2023/html/Tan_Learning_on_Gradients_Generalized_Artifacts_Representation_for_GAN-Generated_Images_Detection_CVPR_2023_paper.html,
  code https://github.com/chuangchuangtan/LGrad). Converts images to gradients w.r.t. a
  pretrained CNN and classifies those. Reported gain 11.4% over prior SOTA. Requires
  training a classifier; the transform model is pretrained.
- **NPR** — Tan et al., *Rethinking the Up-Sampling Operations in CNN-based Generative
  Network for Generalizable Deepfake Detection*, CVPR 2024, arXiv:2312.10461
  (https://arxiv.org/abs/2312.10461). Neighbouring-Pixel Relationships: a local,
  relative-difference representation of up-sampling artefacts, in *image* space rather than
  frequency space. Trained on ProGAN, evaluated on 28 generation techniques; reported +12.8%
  over prior methods. **But it scores 0.53 AUC on Deepfake-Eval-2024** (§1.2).

**The relevant warning for this project's SRM/FFT channels:** Aliyev & Rustamov, *Data
Diversity, Not Frequency Invariance: A Controlled and Self-Audited Study of
Compression-Robust Deepfake Detection*, arXiv:2608.28685, August 2026
(https://arxiv.org/html/2608.28685). They pre-registered a stop-gate, ran a four-pass
adversarial self-audit, and found a **plain EfficientNet-B0 with no frequency streams beat
their frequency-aware model**: 0.9590 vs 0.9224 video-AUC at c40 compression (Δ +0.0366,
CI [+0.020, +0.054]). Under a matched training recipe the frequency streams added **no
detectable difference** (Δ −0.0004, CI [−0.0092, +0.0080]). What *did* help was data
diversity: training on real H.264 CRF variants gave 0.963 AUC vs 0.890 for synthetic JPEG
augmentation, a +7.3 point gap.

This is the same null result this project already found for SRM, FFT, CBAM and GeM — and it
is a *published* null result, which is the point.

### 2.4 Reconstruction-error methods

- **DIRE** — Wang, Bao, Zhou, Wang, Hu, Chen, Li, *DIRE for Diffusion-Generated Image
  Detection*, arXiv:2303.09295 (https://arxiv.org/abs/2303.09295). Reconstruct the image
  through a pretrained diffusion model and measure the error; generated images reconstruct
  better. Generalises across *diffusion* models. Needs a diffusion model at inference —
  expensive. Its distilled variant scores **0.52 AUC** in the wild (§1.2).
- **AEROBLADE** — Ricker, Lukovnikov, Fischer, *AEROBLADE: Training-Free Detection of Latent
  Diffusion Images Using Autoencoder Reconstruction Error*, CVPR 2024, arXiv:2401.17879
  (https://arxiv.org/abs/2401.17879). **Training-free.** Uses only the LDM autoencoder;
  generated images reconstruct with lower error. Nearly matches trained detectors, and can
  also highlight inpainted regions. Scope: latent diffusion only.

### 2.5 Patch and local-texture

- **PatchCraft** — Zhong, Xu, Li, Qian, Zhang, *PatchCraft: Exploring Texture Patch for
  Efficient AI-generated Image Detection*, arXiv:2311.12397
  (https://arxiv.org/abs/2311.12397). "Smash & Reconstruction" erases global semantics and
  amplifies texture; classifies inter-pixel correlation contrast between rich- and
  poor-texture regions. Benchmarked over 17 generators. Independently measured at 67.5% mean
  accuracy zero-shot (§1.3) — third best of 16.
- **LaDeDa / Tiny-LaDeDa** — Cavia, Horwitz, Reiss, Hoshen, *Real-Time Deepfake Detection in
  the Real-World*, arXiv:2406.09398 (https://arxiv.org/abs/2406.09398). A classifier over
  **9×9 patches**, image score = pooled patch scores. ~99% mAP in-domain, 93.7% mAP on
  WildRF. Tiny-LaDeDa is **4 convolutional layers**, 375× fewer FLOPs, 10,000× fewer
  parameters. Genuinely laptop-scale.
- **SAFE** — Li, Cai, Hao, Jiang, Hu, Feng, *Improving Synthetic Image Detection Towards
  Generalization: An Image Transformation Perspective*, KDD 2025, arXiv:2408.06741
  (https://arxiv.org/abs/2408.06741, code https://github.com/Ouxiang-Li/SAFE). **This is the
  most directly actionable paper for this project.** Its central move: **replace
  down-sampling with cropping** in preprocessing, because resizing distorts exactly the
  artefacts you want; plus ColorJitter and RandomRotation to kill colour discrepancies, plus
  patch-based random masking. Reported +4.5% accuracy, +2.9% AP over prior work.

  Read that against this project's verified finding that *source native resolution alone
  separates corpora by ~4σ in post-resize high-frequency energy*. Resizing every corpus to
  224×224 **encodes the downsample ratio into the image**. Random-cropping at native
  resolution removes that channel entirely. This is a one-line change to
  `preprocessing.py` with a real, cited justification.

### 2.6 Newer: preserve the pretrained space instead of overwriting it

**Effort** — Yan et al., *Orthogonal Subspace Decomposition for Generalizable AI-Generated
Image Detection*, **ICML 2025 Oral**, arXiv:2411.15633
(https://arxiv.org/abs/2411.15633, code https://github.com/YZY-stack/Effort-AIGI-Detection).
Mechanism: SVD-decompose CLIP's weights, **keep the principal semantic subspace frozen**, and
learn forgery evidence only in its orthogonal complement — so fine-tuning cannot collapse the
pretrained representation. Their words: naively fine-tuning a VFM "risks distorting the
original rich representation feature space," pushing it "to become low-ranked again."

**0.19M trainable parameters** — roughly 1,000× fewer than LSDA (133M) or ProDet (96M).
Numbers on UniversalFakeDetect: linear probe 81.02 mAcc / 90.14 mAP; full fine-tuning 86.22
mAcc / 97.95 mAP; Effort **95.19 mAcc / 99.41 mAP**. On GenImage: full FT 86.22 → Effort
**91.1 mAcc**.

**RINE** — Koutlis & Papadopoulos, *Leveraging Representations from Intermediate
Encoder-blocks for Synthetic Image Detection*, ECCV 2024, arXiv:2402.19091
(https://arxiv.org/abs/2402.19091, code https://github.com/mever-team/rine). Frozen CLIP,
but reads *intermediate* transformer blocks and learns a small module that weights each
block's contribution. **+10.6% average absolute improvement across 20 test datasets**, and —
the number that matters here — "the best performing models require just a **single epoch for
training (~8 minutes)**."

**SSAFE** — Lee, Kim, Nam, Lee, Shin, *SSAFE: Simple and Strong AI-Generated Image Detection
via Frozen Vision Encoders*, arXiv:2606.08634, June 2026
(https://arxiv.org/abs/2606.08634). Frozen PE-Core-G14-448 (also tests CLIP, SigLIP, DINOv2,
DINOv3) + a **linear classifier**, trained on **10K curated images** (vs 288K in AIGIBench,
4M in OpenFake). Results: AIGIBench 89.4% acc / 95.7 AP; AIGI-Holmes 99.9% acc / 100 AP;
OpenFake 98.3% mean TPR / 99.9 ROC-AUC; their own RealWorldBench 98.3% TNR / 94.4% TPR /
99.0 AUC. They report the linear probe on frozen PE-Core outperforming most fine-tuned
baselines including UnivFD and C2P-CLIP.

### 2.7 Ranked by cost, for one laptop GPU

| Method | Trains what | Feasible on an RTX 4060 Laptop? |
|---|---|---|
| AEROBLADE | nothing | yes (inference only, LDM-scope only) |
| UnivFD linear probe | 768→1 linear layer | **trivially** — minutes |
| SSAFE-style frozen-encoder probe | linear layer | **trivially**, once features are cached |
| RINE | small module on frozen CLIP | yes — ~8 min/epoch reported |
| Effort | 0.19M params on frozen CLIP | yes |
| Tiny-LaDeDa | 4 conv layers | yes |
| SAFE / NPR / LGrad / PatchCraft | full backbone | slow but possible |
| DIRE | diffusion inversion per image | no |
| Community Forensics | ViT-S on 2.7M images | no |

---

## 3. The frozen-feature finding — verified, but weaker than the strong form

**The strong version of the claim ("a linear probe on frozen CLIP beats a fine-tuned CNN
because fine-tuning latches onto shortcuts") is directionally right but is not what the
numbers say.** Report it honestly.

### 3.1 What is solidly established

**Kumar, Raghunathan, Jones, Ma, Liang**, *Fine-Tuning can Distort Pretrained Features and
Underperform Out-of-Distribution*, **ICLR 2022 (Oral)**, arXiv:2202.10054
(https://arxiv.org/abs/2202.10054). This is the general theorem, and it is exactly the
mechanism this project is suffering from. Across 10 distribution-shift datasets:

> "fine-tuning obtains on average **2% higher accuracy ID but 7% lower accuracy OOD** than
> linear probing"

They prove it happens because while the head is being learned the lower layers move and
**distort the pretrained features**. Their fix, LP-FT (linear-probe first, then fine-tune),
gets "1% better ID, **10% better OOD** than full fine-tuning."

This is the single most useful citation for the thesis. It says: *fine-tuning ConvNeXt
end-to-end on a corpus with a shortcut is the worst possible choice, and there is a proven,
cheap alternative.*

### 3.2 What UnivFD actually demonstrated

UnivFD (§2.2) is real and the effect is large: on unseen diffusion/AR models, linear probe on
frozen CLIP gets ~95.00 mAP vs ~75.51 mAP for a fully-trained ResNet-50. But note the
confound: that comparison changes **both** the backbone (ViT-L/14 vs ResNet-50) **and** the
training regime (frozen vs full). It is not a clean linear-probe-vs-fine-tune ablation on a
fixed backbone. The paper itself has **no direct LP-vs-FT comparison on the same backbone**.

### 3.3 Where the strong claim breaks

Two papers that hold the backbone fixed and vary the adaptation say plain linear probing is
**not** the ceiling:

- **Effort** (§2.6), on UniversalFakeDetect: linear probe on frozen CLIP **81.02 mAcc /
  90.14 mAP**; **full fine-tuning 86.22 mAcc / 97.95 mAP**. Full fine-tuning *beats* the
  linear probe here. Their point is not "freeze everything" but "adapt in a subspace that
  cannot destroy the pretrained one" (95.19 mAcc).
- **Yermakov, Čech, Matas**, *Unlocking the Hidden Potential of CLIP in Generalizable
  Deepfake Detection*, arXiv:2503.19683, March 2025 (https://arxiv.org/abs/2503.19683).
  Same frozen CLIP, cross-dataset face deepfake AUROC:

  | Adaptation | CDFv2 | DFD | DFDC | FFIW | DSv1 |
  |---|---|---|---|---|---|
  | Linear probing | 78.13 | 88.15 | 73.62 | 79.28 | 63.65 |
  | LN-tuning | 94.88 | 96.83 | 86.41 | 92.24 | 83.57 |
  | LN-tuning + Norm | 96.21 | 98.18 | 87.82 | 92.72 | 88.81 |
  | Full method | 96.62 | 98.00 | 87.15 | 91.52 | 92.01 |

  Their conclusion is "PEFT is better for generalization than full fine-tuning in low data
  regimes" — not that linear probing wins. They also note LoRA hit 99.99% *training* AUROC in
  one epoch, i.e. it overfits instantly.

### 3.4 And the frozen features are not future-proof either

UnivFD's own frozen-CLIP linear probe measures **40.7% mean accuracy** zero-shot in the 2026
independent benchmark (§1.3), **52.4% balanced accuracy on Synthbuster** (§4.2), and **0.56
AUC in the wild** (§1.2). A 2023 linear probe does not detect 2026 generators.

### 3.5 The honest synthesis, and what to do

The defensible statement is:

> Adapting a strong pretrained representation **with very few trainable parameters**
> generalises far better out-of-distribution than fine-tuning a whole backbone on a
> shortcut-laden corpus. Plain linear probing is the cheapest point on that curve and a
> strong baseline, but parameter-efficient adaptation (LN-tuning, orthogonal-subspace, small
> block-weighting modules) beats it, and *recency and diversity of training data* dominate
> both.

For this project that is still the right pivot, for three reasons that are independent of
whether LP beats PEFT:

1. It is **minutes of compute**. Cache CLIP/DINOv2 features for 78k images once (a single
   forward pass — on an RTX 4060 expect roughly 20–40 minutes for ViT-L/14 at 224px, my own
   estimate, not from a paper), then `sklearn.linear_model.LogisticRegression` fits in
   seconds. You can run a full leave-one-source-out sweep — 20 folds — in an afternoon.
   Fine-tuning ConvNeXt 20 times is a week.
2. A ~1k-parameter classifier **cannot memorise 20 corpus signatures** the way 50M
   fine-tuned parameters can. It is the natural control condition for a shortcut hypothesis.
3. It gives a clean, cheap, three-way comparison for the thesis: frozen probe vs PEFT vs the
   existing fully fine-tuned ConvNeXt, **all under leave-one-source-out**. If the frozen probe
   loses in-domain and wins cross-source, that is Kumar et al.'s result reproduced on a novel
   dataset. That is a thesis chapter.

Implementation, concretely: `open_clip` or `transformers` (neither is currently installed in
`.venv`), CLIP ViT-L/14 image encoder, take the 768-d pooled feature, L2-normalise,
`LogisticRegression(max_iter=1000)` — scikit-learn 1.8.0 is already installed in
`venv-linux`. Ojha et al.'s repo is the reference implementation.

---

## 4. Shortcut and bias literature for this exact problem

### 4.1 The theoretical frame

**Geirhos, Jacobsen, Michaelis, Zemel, Brendel, Bethge, Wichmann**, *Shortcut learning in deep
neural networks*, **Nature Machine Intelligence 2(11):665–673**, 10 Nov 2020,
DOI 10.1038/s42256-020-00257-z, arXiv:2004.07780 (https://arxiv.org/abs/2004.07780). Venue,
volume, pages and DOI confirmed via Crossref and the arXiv page. Shortcuts are "decision rules
that perform well on standard benchmarks but fail to transfer to more challenging testing
conditions, such as real-world scenarios." The prescribed diagnostic is **out-of-distribution
testing** — which is exactly what leave-one-source-out is.

### 4.2 The corpus→class confound, documented

This confound is now a named, active research topic. The closest papers:

- **Zheng et al., *Breaking Semantic Artifacts for Generalized AI-generated Image Detection*,
  NeurIPS 2024.** (Proceedings page and slides verified; **no confirmed arXiv ID — cite the
  proceedings URL.**) Detectors "suffer from substantial Accuracy drops in such cross-scene
  generalization"; the authors attribute the failure to **"semantic artifacts" in both real
  and generated images**, noting that "real images with different semantics exhibit different
  artifacts" and that "semantic artifacts can be inherited by generative models." Their method
  gains +2.08pp cross-scene and +10.59pp over all 31 test sets. *(A widely-quoted "51.18%
  Bedroom→Church" figure is **unverified** — snippet only.)*

- **B-Free** — Guillaro, Zingarini, Usman, Sud, Cozzolino, Verdoliva, *A Bias-Free Training
  Paradigm for More General AI-generated Image Detection*, arXiv:2412.17671
  (https://arxiv.org/abs/2412.17671). **Venue unverified — no acceptance note on arXiv.**
  Identifies "spurious correlations such as **content, format, or resolution**."

  The constructive fix, and the structural answer to "every corpus is one class":
  **generate the fakes from the reals.** 51,517 real MS-COCO images regenerated by SD-2.1
  empty-mask inpainting → 309,102 fakes, "ensuring semantic alignment between real and fake
  images, allowing any differences to stem solely from the subtle artifacts introduced by AI
  generation." Content-based augmentation adds same- and cross-category inpainting, scaling,
  cut-out, noise, jitter.

  **Their Figure 2 is the single most quotable result for this thesis: the same detector
  (RINE) gives opposite predictions on the same generator (DALL·E 3) depending on whether it
  was trained with RAISE or COCO as the real corpus.** That is the confound, isolated.
  Results: 96.3 vs 67.3 average balanced accuracy across 27 generators; on Synthbuster,
  B-Free 99.6% balanced accuracy vs RINE 54.6% and UnivFD 52.4%; on DALL·E 3, B-Free 98.2%
  vs UnivFD 47.3%, RINE 45.3%.

- **SFLD / TwinSynths** — Gye, Ko, Shon, Kwon, Kim, *SFLD: Reducing the content bias for
  AI-generated Image Detection*, **WACV 2025 (Oral)**, arXiv:2502.17105
  (https://arxiv.org/abs/2502.17105). Contributes TwinSynths: "visually near-identical pairs
  of real and synthetic images", i.e. content held constant across the class boundary.

- **SynthCLIC** — Willi, Mathys, Graber, arXiv:2602.12381. Caption-matched real/fake pairs.
  CLIP linear detectors: 0.96 mAP on GAN-heavy data, 0.92 on SynthCLIC, but **0.42 mAP**
  transferring to CNNSpot. Their cue analysis is a description of corpus identity in words:
  high synthetic scores track "cleaner, more compositionally controlled, technically polished"
  images; low scores track "messier capture conditions and provenance cues characteristic of
  real photographs." That is precisely the failure mode reported for arbitrary user photos.

- **Shuai et al.**, arXiv:2603.09242 (Mar 2026) — names the mechanism **"semantic fallback"**.
  **Zhang et al.**, arXiv:2604.12353 (Apr 2026) — separates "generation-pattern bias" from
  "content bias", +10.89% accuracy.

**Where the confound comes from.** ForenSynths (Wang et al., CVPR 2020, arXiv:1912.11035)
trains on 20 LSUN categories against per-category ProGAN, and tests on 13 generators each
carrying its own real corpus. GenImage pairs ImageNet **JPEGs** against generator **PNGs**.
The useful contrast is FF++ (Rössler et al., **ICCV 2019, pp. 1–11, DOI
10.1109/ICCV.2019.00009**, Crossref-verified): its manipulations derive from the same 1,000
source videos, so it does **not** have this confound — its shortcut is compression level
instead.

*(Note: the "Preserving Fairness Generalization" line — Lin et al., CVPR 2024,
arXiv:2402.17229 — is demographic fairness, not corpus identity. Adjacent; do not conflate.)*

### 4.3 JPEG, resolution, resampling and post-processing as shortcuts

**Grommelt, Weiss, Pfreundt, Keuper**, *Fake or JPEG? Revealing Common Biases in Generated
Image Detection Datasets*, **ECCV 2024 Workshops, LNCS 15644, pp. 80–95, 2025**,
DOI 10.1007/978-3-031-92089-9_6, arXiv:2403.17608 (https://arxiv.org/abs/2403.17608).
Crossref-verified.

- GenImage reals are ImageNet JPEGs, "the majority... compressed using a quality factor of
  96"; fakes are uncompressed PNG.
- Fakes have **one fixed size per generator**: 1024² Midjourney, 512² SD1.4/1.5/Wukong,
  256² GLIDE/ADM/VQDM, 128² BigGAN. **This is the same structure as this project's verified 4σ
  resolution separation.**
- Debiasing: keep only QF-96 reals, JPEG-96 the fakes, restrict size to [450,550] px — which
  cuts training from ~320k to 75k images.
- **Cross-generator accuracy: ResNet-50 71.68% → 82.74% (+11.06pp); Swin-T 74.09% → 85.83%
  (+11.74pp).** JPEG robustness gains +13.26 / +8.75 / +4.49 points at QF 95/80/60.
- They do **not** run a metadata-only classifier. See §4.6.

**Gragnaniello, Cozzolino, Marra, Poggi, Verdoliva**, ICME 2021, arXiv:2104.02617
(https://arxiv.org/abs/2104.02617). Table 3 verified from the PDF. The Wang2020
ProGAN-trained detector, accuracy / Pd@5%: StyleGAN2 71.5/69.0, BigGAN 59.2/45.2, RelGAN
63.6/56.0 — against a variant **without the first-layer downsampling** at 92.2/88.8,
93.5/92.0, 92.8/86.6. Their conclusion: **"a 2× downsampling has catastrophic effects."**

That is a directly cited, quantified reason why this project's 224×224 resize is destroying
the forensic signal while simultaneously encoding the corpus identity.

Their protocol discipline is worth copying wholesale: a matched harness ("all networks are
trained and tested on the very same data"), per-detector preprocessing recorded as an explicit
variable, **Pd@5% / Pd@1% FAR reported because AUC hides the failure**, and CelebA-HQ reals
excluded "since they are GAN-upsampled versions of the low-resolution real images."

**Corvi, Cozzolino, Zingarini, Poggi, Nagano, Verdoliva**, *On the detection of synthetic
images generated by diffusion models*, **ICASSP 2023, pp. 1–5, DOI
10.1109/ICASSP49357.2023.10095167**, arXiv:2211.00680 (https://arxiv.org/abs/2211.00680).
Social-network degradation modelled as random large crop → resize to 200×200 → JPEG QF
65–100. On DALL·E 2: Grag2021 **94.9 → 64.4 AUC**; Wang2020 **85.8 → 44.8 AUC**, i.e. below
chance.

**Corvi et al.**, *Intriguing properties of synthetic images: from generative adversarial
networks to diffusion models*, **CVPRW 2023 (WMF), pp. 973–982, DOI
10.1109/CVPRW59228.2023.00104**, arXiv:2304.06408. Their finding (2) is the citation for
corpus inheritance: "when the dataset used to train the model lacks sufficient variety, its
biases can be transferred to the generated images."

**BIAS-ID** — Ricker, Fischer, Quiring, arXiv:2605.31153 (May 2026). An audit instrument:
six detectors × two datasets (Synthbuster, SynthCLIC) × five transforms, scored by **score
shift** and **Aggregated Transform Sensitivity**. JPEG ATS: AIDE −0.547 / −0.528; B-Free
0.045 / −0.093 (least biased). WebP: DRCT +0.549 on reals. Grayscale: SPAI −0.863 on fakes.

**Zhou & Wang**, *How Fragile Are Training-Free AI-Generated Image Detectors? A Controlled
Audit of Score Direction, Preprocessing, and Compression*, arXiv:2606.20488 (Jun 2026).
Abstract-verified numbers: swapping the LPIPS backbone AlexNet→VGG-16 moves AUROC by
**+0.085**; **resize-512 vs native preprocessing flips per-generator conclusions by up to
0.38 AUROC**; a RIGID-style score *inverts* (AUROC < 0.5) at σ=0.05 and collapses to 0.15 at
σ=0.3; and "without unified re-encoding, AUROC under JPEG-50 **exceeds** the clean condition."

**Aliyev & Rustamov**, *Data Diversity, Not Frequency Invariance*, arXiv:2608.28685 (Aug 2026,
preprint submitted to IEEE Access). See §2.3 — pre-registered stop-gate, capacity- and
augmentation-matched controls, and a **published self-audit of their own negative result**.
Plain EfficientNet-B0 beats the frequency-invariance architecture by 3.66 AUC at CRF 40 on
FF++; codec-diverse training data beats synthetic JPEG augmentation by 7.3 points.

**Ricker, Damm, Holz, Fischer**, *Towards the Detection of Diffusion Model Deepfakes*,
VISAPP 2024, arXiv:2210.14571 — GAN detectors fail on diffusion images, near-perfect after
retraining.

### 4.4 The protocols careful papers actually use

**The most complete protocol specification found is in the video domain, and it transfers
directly.** Cakiroglu, Lu, Dalkilic, Kurban, *Auditing Generalization in AI-Generated Video
Detection: A Six-Control Protocol and the VidAudit Toolkit*, arXiv:2606.31004 (Jun 2026).
It contains the existence proof this thesis needs: **a three-feature clip-length classifier
reaches leave-one-generator-out AUC 0.998 on GenVidBench "while measuring nothing about
motion"**, and a 20-paper survey found **none** applying all six controls.

The six controls, quoted:

1. **C1 canonical re-encode** — "every input passes through one H.264 pipeline before feature
   extraction". *(Image analogue: re-encode every image to one JPEG quality and one size
   before anything else.)*
2. **C2 leakage-audited filter** — "a trivial-baseline classifier... is fit on the
   uncontrolled pipeline to measure an upper bound on leakage".
3. **C3 real-vs-real coherence probe** — "a classifier is trained to discriminate the two real
   sources, giving a measurable floor on dataset-bias contribution".
4. **C4 matched-harness comparison** — the same readout (L2 logistic regression) for every
   detector.
5. **C5 multi-seed stability + bootstrap confidence intervals.**
6. **C6 cross-dataset validation** on a separately curated benchmark.

**C3 is this project's confound, operationalised.** Their numbers: the trivial classifier
drops 0.998 → **0.529** once controlled; CLIP scores 0.852 against a real-vs-real floor of
0.766 (margin +0.086) — "caught carrying dataset identity"; WaveRep scores 0.996 against a
floor of 0.534. At FPR 0.1% several high-AUC methods fall to ≤0.03 recall and **the
leaderboard reorders**.

**Standard cross-generator tables:**
- **ForenSynths / CNNDetection**: train on ProGAN + LSUN, test on 13 generators.
- **GenImage**: train on one of eight generators, test on the other seven; plus a
  degraded-image task.
- **DF40** (Yan et al., **NeurIPS 2024 Datasets & Benchmarks**, arXiv:2406.13495) — "4
  standard evaluation protocols"; **the four are named only in the body — unverified.**
- **Cross-dataset face deepfakes**: train FF++, test Celeb-DF-v2 / DFD / DFDC / FFIW /
  DeepSpeak (see the Yermakov table, §3.3).

**On "who defines leave-one-generator-out": no originating paper could be attributed.**
Multiple papers use the term with a consistent definition. Cite a paper that reports the
table (VidAudit gives a worked, audited LOGO table); do not claim an originator.

**Protocols that enforce same-source real/fake — the actual fix:**
- **B-Free** — regenerate the reals themselves (§4.2).
- **TwinSynths / SFLD** — near-identical real/synthetic pairs.
- **SynthCLIC** — caption-matched pairs.
- **FakeInversion** — Cazenavette, Sud, Leung, Usman, **CVPR 2024**, arXiv:2406.08603. Uses
  reverse image search to "mitigate stylistic and thematic biases in the detector evaluation";
  the resulting scores "align well with detectors' in-the-wild performance."
- **INP-X** — real / inpainted / pixel-exchanged triplets (§5.2).
- *(MS COCOAI, arXiv:2601.00553, is built on COCO with five generators, but its claimed
  caption-level alignment is **unverified**.)*

**Protocols that hold JPEG/resolution constant:** Grommelt's debiased GenImage; the "GenImage
unbiased" split B-Free tests on (5k/5k, matched JPEG); Gragnaniello's fixed per-detector
preprocessing; VidAudit C1; SAFE's crop-instead-of-resize (§2.5).

**Other 2025–26 protocol papers:**
- **AI-GenBench** — Pellegrini et al., **IJCNN 2025 (Verimedia workshop)**, arXiv:2504.20865.
  Temporal/chronological incremental protocol, explicitly against "arbitrary dataset splits,
  unfair comparisons, and excessive computational demands."
- **NTIRE 2026 Challenge** — Gushchin et al. (54 authors), CVPR 2026 workshop,
  arXiv:2604.11487. 42 generators, **36 transformations**, AUC scored over transformed and
  clean together.
- **RRDataset** — Li et al., **ICCV 2025**, arXiv:2509.09172. Scenario / transmission /
  re-digitization axes.
- **Michels et al.**, arXiv:2607.00948 — motion-based video detectors "collapse to
  near-random levels" once the motion bias is removed. Same shape of result as this project's
  augmentation-monotonicity finding.

### 4.5 The number that most directly indicts architecture work

**Ren et al.**, arXiv:2602.07814 (§1.3): **"training data alignment critically impacts
generalization, causing up to 20–60% performance variance within architecturally identical
detector families."** The corpus decides the result, not the architecture. Ranking instability
across dataset pairs: Spearman ρ 0.01–0.87.

### 4.6 Two gaps in the image-domain literature — both are cheap and both are available

These came out of the protocol survey and are directly relevant to what this project could
contribute:

1. **No image-domain paper trains a classifier on JPEG-quality / image-size metadata alone
   as a leakage probe.** Grommelt et al. document the bias but never run it. This is VidAudit
   control C2 with no published image-domain instance.
2. **No image-domain real-vs-real coherence probe.** Training a classifier to separate COCO
   vs LSUN vs RAISE vs Places365 vs FFHQ vs OpenImages, and publishing that AUC as **the
   floor below which no detection claim is meaningful**, appears unclaimed. This is VidAudit
   control C3 with no published image-domain instance.

This project has **six real corpora and twenty sources sitting on disk already**. Both probes
are logistic regressions. See §7.

## 5. The 3-class formulation

**Bottom line: the label space unifies cleanly; the feature space does not.** The 3-class
formulation is real, current and published — but *every* paper that uses it pairs the
"edited" class with a **pixel-level mask head**, and every paper that drops that head reports
near-total failure on that class. The recommendation is therefore **not** "split into two
models" and **not** "stay as you are": it is **keep the 3-class label space, add a
segmentation branch, and get off 224×224.**

### 5.1 The formulation exists, verbatim

- **SIDA / SID-Set** — CVPR 2025, arXiv:2412.04292. 300K images: "100K real, 100K synthetic,
  and 100K tampered."
- **So-Fake-Set** — arXiv:2505.18660. "A unified three-way protocol over REAL, FULL SYNTHETIC,
  and TAMPERED."
- **Gallina et al., *From Detection to Localization***, **ACM MM 2026 DFF workshop**,
  arXiv:2609.02640 (submitted six days ago). Literally "a unified multiclass framework
  (real vs. fully generated vs. tampered)" **plus a segmentation branch**.
- **HiFi-IFDL** — Guo, Liu, Ren, Grosz, Masi, Liu, *Hierarchical Fine-Grained Image Forgery
  Detection and Localization*, **CVPR 2023, pp. 3155–3165**, arXiv:2303.17111. Level 1 of its
  hierarchy **is** Fully-synthesized vs Partial-manipulated. Cost: 1.71M training images,
  400K iterations at batch 16, high-resolution masks on every image. Its own stated limitation
  is that it generalises **poorly on diffusion-based inpainting**, with the failure mode named
  as "inpainted images have small forgery regions."

So the answer to "is 3-class supported?" is yes — with a mask head attached, and at
resolutions this project is not using.

### 5.2 The separate-task view, stated by the unifiers themselves

- **ForensicHub** — NeurIPS 2025 Datasets & Benchmarks, arXiv:2505.11003: the field has
  "domain silos, where each domain independently constructs its datasets, models, and
  evaluation protocols."
- **SICA** — arXiv:2602.06676 — measures the transfer directly: **AIGC→IMDL AUC 0.4161,
  IMDL→AIGC 0.4706 — both below chance.** Naive unified training *drops* AIGC performance
  from 0.9291 to 0.8987. Features genuinely do not transfer between the two tasks.
- Manipulation-localisation features are SRM / BayarConv noise residuals, DCT-JPEG traces,
  Sobel boundary cues, PRNU fingerprints. TruFor's Noiseprint++ is trained **on real images
  only** and detects forgery as *deviation from a camera model* — structurally blind to a
  fully-generated image, which has no camera model to deviate from.

This is the reason a single flat softmax over three classes cannot work on shared features
alone: the evidence for class 2 and the evidence for class 3 live in different places.

### 5.3 Image-level "edited" at 224×224 is not learnable — four independent measurements

This is the decisive evidence for this project.

1. **SIDA Table 2**, per-class accuracy on SID-Set, baselines retrained at crop 224:
   Gram-Net **0.8%**, Fusing **2.7%**, LNP **2.9%**, LGrad **6.8%**, CNNSpot **6.9%** on the
   **tampered** class — while the *same models* score 83–94% on the fully-synthetic class.
   SIDA itself, at 1024×1024 with a mask branch, gets **92.7%**.
2. **IML-ViT Table 9**: resizing to 512 instead of padding to 1024 costs **14.5 points of mean
   pixel F1** (0.5605 → 0.4153) — the largest single ablation in the table, larger than
   removing edge supervision. And that is 512, not 224. Their reason: "their resizing methods
   already harm these first-hand artifacts." Their Table 2 gives the reason it matters:
   CASIAv2 averages **7.6% tampered pixels; DEFACTO only 1.7%.** At 224×224, 1.7% of the
   image is about 850 pixels.
3. **Gallina et al. Table 3**, same 3-class head, backbone swapped:
   CLIP @224 → Real 95.1 / Fake 95.9 / **Tampered 69.0**;
   DINOv2 @518 → Real 93.1 / Fake 99.8 / **Tampered 93.3**.
   The +24.3 points land almost entirely on the tampered class. *(Caveat: backbone and input
   resolution change together, so this is not a clean resolution ablation.)*
4. **TGIF** — Mareen et al., **IEEE WIFS 2024**, arXiv:2407.11566, states it plainly:
   "We have verified that, indeed, **no SID method is able to detect the synthetic spliced
   regions**." And conversely, no image-forgery-localisation method localises fully-regenerated
   inpainting — CAT-Net goes from 0.87 F1 on spliced regions to **0.04** on fully-regenerated
   ones. CAT-Net also drops 0.87 → **0.05** at JPEG Q80.
5. **ForensicHub Table 7**, everything at 256×256 image-level: GenImage 0.99–1.00 for nearly
   every model, while Columbia sits at 0.298 (MVSS-Net), 0.306 (TruFor), 0.285 (Mesorch),
   0.199 (FatFormer) — **below chance**.

### 5.4 And the global-artefact shortcut on top of that

**Nebioglu, Bilgiç, Popescu**, *AI-Generated Image Detectors Overrely on Global Artifacts:
Evidence from Inpainting Exchange*, arXiv:2602.00192, January 2026
(https://arxiv.org/html/2602.00192). INP-X takes an inpainted image and **surgically restores
the original pixels outside the mask**, keeping the generated content inside it. 90K matched
triplets (real / inpainted / exchanged) over 4 datasets and 3 inpainting models:

| Detector | Standard inpainting | INP-X |
|---|---|---|
| Corvi2023 (frequency-based) | 94.2% | **55.4%** |
| DNF | 71.0% | 60.4% |
| CLIP 10+ | 65.8% | 56.3% |
| HiveModeration (commercial) | 91.4% | **54.8%** |
| Sightengine (commercial) | 92.6% | **55.0%** |

Across 11 open-source detectors, best INP-X accuracy is **60.4%**; several are at chance.
Mechanism: latent-diffusion inpainting passes the **whole image** through the VAE
encoder-decoder, leaving "a subtle yet pervasive spectral shift across the entire image,
including unedited regions." Detectors read that global re-encode, not the edit. Their
conclusion: reported >90% accuracy on inpainting benchmarks "appears misleading, reflecting
detector reliance on trivial shortcuts."

### 5.5 What the standard forgery datasets actually expect

- **DEFACTO** — Mahfoudi, Tajini, Retraint, Morain-Nicolier, Dugelay, Pic, **EUSIPCO 2019**.
  Built on **MSCOCO segmentations** refined by alpha matting. 19K copy-move, 25K inpainting,
  105K splicing, 80K morphing. Ships a **probe mask, donor mask, inpaint mask and a JSON
  record of every operation**. It is a localisation benchmark.
- **IMD2020** — Novozámský, Mahdian, Saic, **WACV Workshops 2020, pp. 71–80**. 70K synthetic
  + 2K real-life manipulations, **official binary masks**. Its Table 1 is a survey of forgery
  datasets with a "Binary mask" column — the field's own framing.
- **CASIA v1/v2** — **ships no official masks.** The community derives them by subtraction
  (Pham et al. 2019); a corrected-groundtruth repository exists because the circulated masks
  are noisy.
- DEFACTO, CASIA, tampCOCO, CocoGlide, TGIF, SID-Set, So-Fake and DiffSeg30k are **all
  COCO-derived** — which is also the cross-source leakage risk flagged in
  `docs/REVIEW_2026-09-08.md` §D.

This project currently discards the masks that DEFACTO and IMD2020 ship, resizes to 224, and
asks for an image-level label. That throws away nearly all the supervision in the data and
asks the model a question the resolution cannot answer.

### 5.6 Recommendation on the formulation

**Keep the 3-class label space. Add a mask head. Raise the resolution. Do not split into two
models.**

Concretely, and this is a single-GPU budget: frozen DINOv2 → shared features → (a) a 3-way
classification head, (b) a light decoder to a binary tamper mask, trained with BCE + Dice.
Gallina et al. reach **IoU 77.8 vs 48.6 prior SOTA with 96.9M parameters, 370 MB, 16.4 ms per
image on a single L40s, and 3+5 epochs.**

Non-negotiables from the evidence above:
- **Minimum 512 px, never 224.** Prefer cropping to resizing; tile-and-vote at test time.
- **Report per-class accuracy and F1, never overall accuracy alone** — overall accuracy is
  exactly what hides a dead class, and SIDA Table 2 shows the tampered class dying at 0.8%
  while the aggregate still looks respectable.
- Report **pixel F1 / AUC / IoU on the tampered class** alongside the image-level numbers.
- Include JPEG and WebP recompression in the evaluation.

**And the warning that applies immediately:** if this project already reports healthy
`ai_edited` accuracy at 224×224, that is itself evidence of a shortcut, because the
literature says it should be near zero. Given `real` is COCO/FFHQ/ImageNet/OpenImages/Places
and `ai_generated` is LSUN/LAION-derived generator output, the model is separating on
semantics and provenance, not artefacts.

---

## 6. Benchmarks and leaderboards active in 2026

### 6.1 FaceForensics++ — server live, leaderboard frozen since October 2022

Verified today, 2026-09-08, by fetching https://kaldir.vc.in.tum.de/faceforensics_benchmark/,
the login and registration forms, the documentation, and the benchmark download.

**Live:** the page renders, the registration and login forms are real server-rendered inputs,
the documentation is up, and the benchmark image set is a direct download (HTTP 200, **565 MB,
1,000 images**). The board is topped by DirechletEnsemble-Classifier at **0.973**, then
Beijing ZKJ 0.941, ZAntiFakeBio 0.940, down to YSNet 0.513.

**Frozen:** by counting table cells across Wayback snapshots, the leaderboard has held at
**exactly 109 entries since 2022-10-06**, and the 2022 snapshot is byte-identical to today's.
There are no dates on the board and no pagination. **`docs/EXTERNAL_BENCHMARK.md`'s "entries
stop around 2024" was optimistic by two years — it is 2021–2022.** *(That file has since been
corrected on disk.)*

**The diagnostic that settles server-vs-community:** ScanNet runs on the same server, the same
codebase, from the same lab (the About page credits the ScanNet authors for the website
sources). Its board went 9,187 → 10,453 cells in the last nine months. **The TUM
infrastructure is demonstrably still scoring in 2026 — FF++ is community-dead, not
server-dead.** Whether the FF++ scorer specifically still returns a result cannot be proven
without spending a submission.

**Rules, verified verbatim:** "we block updates to the test set results of a method for two
weeks after a test set submission." Parameters may be tuned on training data only; the test
set "must only be done once for the final system." Registration is one email address per
person — "We will ban users or domains if required."

**Format, verified by downloading and unzipping the official example:** one JSON, 1,000
entries, `{"0000.png": "fake", ...}`.

Paper: Rössler, Cozzolino, Verdoliva, Riess, Thies, Nießner, **ICCV 2019, pp. 1–11,
DOI 10.1109/ICCV.2019.00009**, arXiv:1901.08971 (https://arxiv.org/abs/1901.08971).

**Verdict:** still worth one zero-shot submission — it is a single inference pass — but go in
accepting that the entry may never appear on a board that has not moved in four years. It is
a sanity check, not a headline.

### 6.2 The one real competition — and it closed three days ago

**NTIRE 2026, "Robust AI-Generated Image Detection in the Wild"** (CVPR 2026 workshop),
Gushchin et al. (54 authors), arXiv:2604.11487. **541 participants, 3,449 submissions,
ROC-AUC scored, 42 generators and 36 transformations.** All four phases verified as
`Previous` via the Codabench API — it **closed 2026-09-05**. It is annual and opened
mid-January, so **calendar early January 2027**. That is the single best externally-comparable
target if the thesis timeline reaches it.

Every other image/video challenge is closed: ACM MM XPlainVerse, IJCAI DDL-X, 1M-Deepfakes,
DFGC (defunct since 2023), Trusted Media Challenge (dead since 2021). The only live deepfake
competition anywhere right now is **RTC-SDD, which is audio.**

**Also: Papers With Code is gone** — paperswithcode.com now redirects to HuggingFace papers.
The "look up the SOTA table" route no longer exists.

### 6.3 Deepfake-Eval-2024 — the strongest currently-available option

arXiv:2503.02857 (§1.2). **20.3 GB, only 1,975 images**, actively maintained (HuggingFace
`lastModified` 2026-08-11), evaluation-only license, with **published baselines your zero-shot
number slots straight against** (UFD 0.56, DistilDIRE 0.52, NPR 0.53 AUC). Its headline —
image detectors lose ~45% AUC in the wild — is the same story a leave-one-source-out result
would tell, which makes it a natural external corroboration.

**Caveat: manually gated.** The access form demands evidence of prior work in the field.
**Apply now**, because the approval latency is the schedule risk, not the compute.

### 6.4 Chameleon — and a correction to the premise

**The paper is not "Are We on the Right Way for Evaluating AI-generated Image Detection?"**
It is **Yan et al., *A Sanity Check for AI-generated Image Detection*, arXiv:2406.19435,
ICLR 2025** (https://arxiv.org/abs/2406.19435). **Chameleon is the dataset; AIDE is the
method.** Distribution is **email-request only** — no HuggingFace release, no evaluation
server.

Its numbers are savage and directly relevant: all ten detectors, **including AIDE**, score
~53–59% mean accuracy, but **near-zero on the fake class** — AIDE gets **0.63% on fakes and
98.46% on reals.** That is the same collapse-to-"everything is real" mode AIGIBench reports
(§1.4), and the same one this project will see on user photographs, mirrored.

### 6.5 WildFake — cite the AAAI version

Two conflicting citations exist: the arXiv preprint (arXiv:2402.11843) lists 2 authors; the
**AAAI-25 camera-ready lists 7 authors and a different title. Cite the AAAI version.**
The dataset is public on ModelScope — verified via their API: **1.17 TB**, 223,558 downloads,
Apache-2.0. The size makes it impractical for a laptop, but subsets are usable.

### 6.6 Offline benchmarks that need no server

These are the realistic ones for a solo student — download, run, report against published
baselines:

- **AIGIBench** — Li et al., NeurIPS 2025 D&B, arXiv:2505.12335,
  https://github.com/HorizonTEL/AIGIBench. 23 fake subsets, four axes (multi-source
  generalisation, degradation robustness, augmentation sensitivity, test-time preprocessing),
  11 published detector baselines. Hardest honest evaluation available offline.
- **AI-GenBench** — Pellegrini et al., IJCNN 2025 (Verimedia workshop), arXiv:2504.20865,
  https://github.com/MI-BioLab/AI-GenBench. Temporal/incremental protocol, and — the reason it
  is on this list — **explicitly designed for practical training requirements**, i.e. built for
  people without clusters.
- **GenImage** — arXiv:2306.08571. No server; a dataset plus the 8×8 cross-generator matrix
  protocol that the thesis's leave-one-source-out table should structurally copy.
- **Synthbuster** — Bammey, IEEE OJSP 2024. 9 generators × 1000 images on RAISE-1K.
  **Already in this project's corpus** — so it can serve as a held-out source at zero data cost.
- **OpenSDI / OpenSDID** — Wang, Huang, Hong, **CVPR 2025**, arXiv:2503.19653,
  https://github.com/iamwangyabin/OpenSDI, CC-BY-4.0. Global synthesis **and** local edits,
  detection **and** localisation. The closest published thing to what this project is trying
  to build, and the model for the two-head formulation in §5.6.
- **WildRF** (LaDeDa, arXiv:2406.09398) and **ITW-SM** (arXiv:2507.10236) — in-the-wild
  social-media sets for a final table row.
- **RRDataset** — Li et al., **ICCV 2025**, arXiv:2509.09172. Scenario / transmission /
  re-digitization axes.

---

## 7. Recommendation, ranked

The premise to accept first: **the 89% cannot be repaired.** It measures corpus recognition.
BACKLOG item 1 says so, `docs/REVIEW_2026-09-08.md` §D shows unmeasured cross-source pHash
leakage on top of it, and §A shows every claimed ablation effect is smaller than the 0.15pp
run-to-run noise floor. There is no version of "improve the model" that fixes a number which
is not measuring the model.

The contribution has to be **the measurement**. Fortunately, the protocol survey turned up two
gaps that are unclaimed, cheap, and sitting on this project's disk.

### Rank 1 — The two leakage probes. Unclaimed in the image domain, and nearly free.

VidAudit (arXiv:2606.31004) defines six controls for auditing generalisation claims. Two of
them **have no published image-domain instance**:

- **C2, the metadata-only leakage probe.** Train a classifier on **JPEG quality factor and
  image dimensions alone** — no pixels — and report its accuracy on the 3-class task. Grommelt
  et al. documented this bias in GenImage but never ran the probe. On this project's 20
  corpora with a verified 4σ resolution separation, this will likely be near-perfect, and
  **that number is the paper.** It is a `pandas` groupby and a logistic regression.
- **C3, the real-vs-real coherence probe.** Train a classifier to separate COCO vs FFHQ vs
  ImageNet vs OpenImages vs Places365 — **all of which carry the same label** — and publish
  that AUC as **the floor below which no detection claim is meaningful.** VidAudit's video
  version caught CLIP at 0.852 against a real-vs-real floor of 0.766, i.e. "carrying dataset
  identity." Nobody has published the image-domain version.

Both are hours of work, both produce a headline number, and together they turn "our model has
a shortcut" into "here is the instrument that measures the shortcut, and here is what it reads
on twenty public corpora." That is a methods contribution, not a failure report.

Do this first. It also tells you whether anything below is worth running.

### Rank 2 — Leave-one-source-out, with the frozen-feature control and a matched harness

1. **Measure the leakage first** (REVIEW §D): pHash over the merged `export_index`, count
   cross-source cross-split pairs at Hamming ≤ 8, with the multi-index banding fix from §D.1.
   DEFACTO, CASIA, tampCOCO and CocoGlide are **all COCO-derived** (§5.5), so a DEFACTO
   inpainting in `test/ai_edited` and its COCO original in `train/real` is contamination
   straddling the exact class boundary the project is about. One pass over the manifest, no
   training. If it is contaminated, every existing number including a future LOSO is affected.
2. Then run **leave-one-source-out over the 20 sources**, with VidAudit C4's **matched
   harness** — the same readout for every arm:
   - the existing fully fine-tuned ConvNeXt-Small,
   - a **frozen CLIP or DINOv2 linear probe** (features cached once, 20 logistic regressions),
   - optionally one PEFT arm (LN-tuning, or Effort-style orthogonal subspace).
3. Report **spread across folds, not just the mean**, in a GenImage-style matrix (§4.4), with
   **per-class** accuracy — never overall alone (§5.6) — plus C5's multi-seed bootstrap CIs.

The frozen arm is what makes this a finding rather than a complaint. Kumar et al. (ICLR 2022)
predicts fine-tuning wins in-domain and loses ~7 points out-of-domain; if that reproduces on a
20-corpus forensic dataset, the thesis has a result with a named mechanism and a theoretical
citation. If it does not reproduce, that is more interesting still.

Compute: one CLIP/DINOv2 forward pass over 78k images (~20–40 min on the RTX 4060 Laptop —
my estimate, not from a paper) plus 20 logistic regressions in seconds. `scikit-learn` 1.8.0
is already installed; `open_clip` / `transformers` is not, and needs adding.

### Rank 3 — Kill the two verified shortcuts, and report what killing them costs

Every fix here is cited and cheap, and the **size of the accuracy drop is the result**:

- **Crop instead of resize** (SAFE, KDD 2025, §2.5). Directly attacks the verified 4σ
  post-resize high-frequency separation. Gragnaniello et al. (ICME 2021, §4.3) quantify the
  other half: **"a 2× downsampling has catastrophic effects"** — 71.5% → 92.2% on StyleGAN2
  from removing one downsampling layer.
- **Equalise JPEG quality and image size across classes** (Grommelt et al., §4.3). They
  measured **+11.06pp (ResNet-50) and +11.74pp (Swin-T)** of cross-generator movement from
  this alone, at the cost of shrinking the training set from ~320k to 75k.
- **Match provenance where the data allows** (B-Free §4.2; BACKLOG 1.2). FF++ and ForgeryNet
  ship paired originals — use each corpus's own originals as its `real` class, for free.
  B-Free's Figure 2 is the citation: **the same detector flips its prediction on the same
  generator depending on whether RAISE or COCO supplied the reals.**

Frame every in-domain drop as a measurement: "intervention X removed N points, therefore at
least N points of the original 89% were shortcut."

### Rank 4 — Fix the formulation: mask head, 512px, per-class metrics

Per §5. The single most likely explanation for a healthy `ai_edited` number at 224×224 is that
it is not real: the literature says that class should be scoring **0.8–6.9%** at that
resolution (SIDA Table 2), and IML-ViT loses **14.5 points of pixel F1** just going from
1024-pad to 512-resize. DEFACTO images average **1.7% tampered pixels**.

Cheapest correct version: frozen DINOv2 → 3-way head + light BCE+Dice mask decoder, ≥512px,
crop not resize, tile-and-vote at test. Gallina et al. hit IoU 77.8 with 96.9M params and 3+5
epochs on one L40s. Report per-class accuracy/F1 **and** pixel F1/IoU on the tampered class.
Cite INP-X (§5.4) for why the old image-level number should not be trusted.

### Rank 5 — Write the ablation null result properly

REVIEW §A already establishes SRM, FFT, CBAM and GeM produce no measurable change (McNemar
p=0.73 for SRM; 0.15pp noise floor from the accidental run-25/26 replicate). That is
publishable, and there is a direct precedent with a protocol to copy: **Aliyev & Rustamov
(arXiv:2608.28685)** published exactly this shape of result for frequency streams — a
pre-registered stop-gate, capacity- and augmentation-matched controls, 10,000-replicate paired
bootstrap, and a **published self-audit that found eight defects in their own experiment, four
of which biased against their hypothesis.**

Two bugs must be fixed before the claim is supportable, both from REVIEW: the **focal-loss
label-smoothing interaction** (§E) and the **batch-dependent FFT normalisation** (§F). Right
now "FFT is useless" is not a supportable claim — only "FFT *as implemented* is useless" is.
Fixing it and re-testing is itself a clean methodology contribution.

### Rank 6 — External calibration

- **Apply for Deepfake-Eval-2024 access today** (§6.3). It is manually gated and the approval
  latency is the schedule risk. Strongest available option: small, maintained, with published
  baselines to slot against.
- **Submit once to FF++** (§6.1). One inference pass, zero-shot, reported with the domain gap
  stated up front. Accept that the board has not moved since 2022.
- **Calendar NTIRE 2027, early January** (§6.2). If the thesis timeline reaches it, that is
  the real leaderboard.
- Failing all of those, run **AIGIBench** or **AI-GenBench** offline (§6.6) — no server, no
  gate, published baselines.

### What not to do

- **Do not chase 89% cross-domain.** Nothing in the literature achieves it (§1.7). Community
  Forensics needed 2.7M images from 4,803 generators to reach 89.2% on unseen models — and an
  independent evaluation put the same method at 78.0%.
- **Do not add another architectural component.** Five have been tested and all sit inside the
  noise floor; the published controlled study found the same for frequency streams against a
  plain EfficientNet-B0 (§2.3); and Ren et al. measure **20–60% performance variance within
  architecturally identical detector families** driven by training data alone (§4.5).
- **Do not report a single 3-way accuracy as the headline** (§5.6) — that is precisely the
  statistic that hides a dead class.

### The one-sentence version

**A rigorous leave-one-source-out study, fronted by the two unclaimed leakage probes
(metadata-only and real-vs-real) and controlled by a frozen-feature arm, is worth far more
than an 89% that measures corpus recognition — and it is the only one of the two that a single
laptop GPU can actually deliver before the deadline.**

---

## Companion documents

Three supporting reviews were produced alongside this one and go deeper on their sections:

- `docs/LIT_SHORTCUT_BIAS.md` — shortcut/bias literature and evaluation protocols (§4)
- `docs/LITERATURE_3CLASS_FORMULATION.md` — the 3-class question in full (§5)
- `docs/BENCHMARK_LIVENESS_2026-09-08.md` — benchmark liveness evidence, incl. the Wayback
  cell counts behind the FF++ finding (§6)

`docs/EXTERNAL_BENCHMARK.md` has been corrected on disk with the verified FF++ status.
