# 2026 datasets and benchmarks for AI-generated / AI-edited image detection

Compiled 2026-09-08. Supersedes nothing; complements `docs/LITERATURE_3CLASS_FORMULATION.md`
(SID-Set / So-Fake) and `docs/BENCHMARK_LIVENESS_2026-09-08.md` (which datasets and
competitions are alive).

Every claim is tagged:

- **[MEASURED]** — I downloaded the actual images and computed the number myself. These are
  the strongest claims in this document.
- **[FETCHED]** — I pulled the paper, repo page or API myself and am quoting it.
- **[SNIPPET]** — a search summary I could not confirm by fetching.
- **UNVERIFIED** — I could not establish it. Do not cite these.

---

## 0. Headline findings

1. **The NTIRE 2026 challenge dataset is the only benchmark found that explicitly engineers
   the confound out.** The organisers generated fakes from captions of their own real images
   *and then aligned the resolution, aspect-ratio and JPEG-quality distributions of the two
   classes on purpose.* It is public, ungated, labels included, and sliceable to a single
   20.6 GB shard. It is binary and ships no masks.
2. **DailyBench (July 2026) is worse than your own dataset on exactly your bug.** I
   downloaded two of its subsets and measured it: every real image is 512×512 and every fake
   is 1024×1024. **Resolution alone classifies it at 100.0% held-out**, against a 58.6%
   baseline. [MEASURED]
3. **So-Fake-Set has the same bug you found.** I pulled one shard and measured it held-out:
   megapixel count alone predicts the 3-class label at **71.1%** against a 43.5% majority
   baseline — 27.6 points of free signal. Real is 67% JPEG; the two fake classes are ~90% PNG.
   [MEASURED]
4. **So-Fake-Set is not too big after all.** Its 2,343 parquet shards are *shuffled* — one
   0.5 GB shard already contains all three classes, masks and 20+ generators. The "1.3 TB"
   figure is the full set, not the minimum usable unit. This is the single most useful
   correction to the current plan.
5. **OpenSDI's binary label decomposes into your exact three classes.** Its `key` field is
   prefixed `entire/` (fully synthetic) or `partial/` (locally edited), reals and fakes all
   come from Megalith-10M, and it is format-clean. Its shards are *not* shuffled, though.
6. Four of the six most promising 2026 localisation datasets (PromptForge-350k, EditSleuth,
   LocateEdit-Bench, SIGMA) exist only as "will be released upon acceptance". They are not
   obtainable today.

---

## 1. 2026 dataset releases

Ordered by usefulness to this project.

### 1.1 NTIRE 2026 Robust AI-Generated Image Detection in the Wild — **the confound-controlled one**

| Field | Value |
|---|---|
| Paper | *NTIRE 2026 Challenge on Robust AI-Generated Image Detection in the Wild* |
| Authors | Aleksandr Gushchin et al. (54 authors) |
| arXiv | **2604.11487**, 2026-04-13 [FETCHED] |
| Venue | CVPR 2026 NTIRE workshop technical report |
| Size | 108,750 real + 185,750 generated = **294,500 images**, 42 generators, 36 transformations [FETCHED, verbatim] |
| Disk | train **114.36 GB** (6 shards), val **4.00 GB**, test-public **0.85 GB** [FETCHED, HF API] |
| Classes | Binary `label ∈ {0,1}` + per-image distortion type and scale |
| Masks | **No** |
| Download | HF `deepfakesMSU/NTIRE-RobustAIGenDetection-train` / `-val` / `-test-public`, all `gated=false` [FETCHED] |

**Why it matters.** Verbatim from the paper [FETCHED]:

> "Generation prompts for this subset are collected from the corresponding real images: we
> first employ a Large Vision-Language model to produce detailed image captions and then
> rewrite them into concise and structured prompts using LLM. **By 'pairing' generated images
> with their real counterparts, we ensure that both subsets reflect similar semantics and
> content distribution, which should help detectors learn content-agnostic features.** To
> further minimize potential biases in generated imagery, we also align its distributions of
> resolutions, aspect ratios, JPEG compression quality factors, and other statistics to those
> of the real subset."

That last sentence is the thing no other dataset in this survey says. Reals come from CC12M,
CommonPool and RedCaps, filtered from 12M down to 100K for train.

Caveat the organisers state themselves: for val and test they use *unpaired* images
deliberately, "to avoid potential advantage from selecting between multiple similar images".
So the pairing is a training-set property.

**Partial download: designed for it.** The train README states "All shards have similar data
distribution, and can be used separately if you prefer to train/test the model on a smaller
set" [FETCHED]. `shard_0.zip` = **20.59 GB** (~50K labelled images). `shard_0 + shard_5` =
**31.96 GB**. Add val (4.00 GB) + test-public (0.85 GB), both with public label CSVs, for a
self-contained ~37 GB package.

**Measured properties of the val split** [MEASURED]: 10,000 images, exactly 5,000 real /
5,000 fake, **every filename is `.jpg` in both classes**, and `is_distorted` is balanced
2,500/2,500 within each class. On a 336-image byte-level sample: 100% JPEG in both classes
with a **single identical quantisation table** (sum 369) throughout, 141 distinct resolutions
among the reals and 131 among the fakes, medians 640,000 vs 562,176 pixels. Best held-out
metadata-only accuracy 60.1% against a 53.0% baseline. See §4.2.

### 1.2 INP-X (Inpainting Exchange) — **small, masked, and it is a confound diagnostic**

| Field | Value |
|---|---|
| Paper | *AI-Generated Image Detectors Overrely on Global Artifacts: Evidence from Inpainting Exchange* |
| Authors | Elif Nebioglu\*, Emirhan Bilgiç\*, Adrian Popescu |
| arXiv | **2602.00192**, 2026-01-30 [FETCHED] |
| Venue | UNVERIFIED (no venue in the arXiv comments field) |
| Size | ~90K images: 20,000 original + 20,000 standard-inpainted + 20,000 exchanged (train); 10,000 + 10,001 + 10,001 (test) |
| Disk | **10.73 GB** [FETCHED, Kaggle API: `totalBytes: 10730785582`] |
| Classes | real / inpainted / *exchanged* (three-way, but the third is an intervention, not a natural class) |
| Masks | **Yes**, first-class `masks/` folder with `CelebAHQ_masks/`, `CityScapes_masks/`, `OpenImages_masks/`, `SUN_RGBD_masks/` plus `mask_sizes.csv` |
| Licence | MIT |
| Download | Kaggle `emirhanbilgic/inpainting-exchange` — `isPrivate: false`, 6,830 downloads, updated 2026-02-09 [FETCHED, API]; code `github.com/emirhanbilgic/INP-X` (HTTP 200) |

**Same source pool: yes, by construction.** Each inpainted image is a local edit of a
specific original from CelebA-HQ / CityScapes / OpenImages / SUN-RGBD, and the "exchange"
variant *restores the original pixels outside the mask*. Inpainters: Kandinsky 2.2,
OpenJourney, SD v1.4.

**Why it belongs in this project specifically.** Its result is the cleanest published
statement of the failure mode you are already fighting: detectors score on a VAE-induced
global spectral shift, not on the synthesised content. When the exchange removes that global
shift, "pretrained state-of-the-art detectors, including commercial ones, exhibit a dramatic
drop in accuracy (e.g., **from 91% to 55%**), frequently approaching chance level"
[FETCHED, abstract].

### 1.3 AUDITS — 530K, masks, two source domains, and a built-in OOD axis

| Field | Value |
|---|---|
| Paper | *Multi-axis Analysis of Image Manipulation Localization*, arXiv **2605.20174**, 2026-05-19 [FETCHED] |
| Size | **529,057 rows** in the released metadata [MEASURED, from the shipped parquet] |
| Disk | **37.2 GB total** — `train.zip` **10.49 GB**, `val.zip` **1.47 GB**, `test.zip` **25.18 GB** [FETCHED, HF API] |
| Classes | Authentic vs manipulated (2-class). `manipulation_type`: Authentic 71,133; GLIDE 78,445; LatentDiffusion 77,769; BlendedDiffusion 62,601; SDXL / BlendedLatentDiffusion / GLIGEN / PowerPaint / StableDiffusion 47,802 each; Firefly 99 [MEASURED] |
| Masks | **Yes** — "All the manipulated images are accompanied by ground-truth masks" [FETCHED] |
| Download | HF `DivyaApp/AUDITS`, **`gated=false`**, MIT licence [FETCHED, API] |

**Same source pool: yes.** Two subsets, both with pristine and manipulated images drawn from
the same corpus: `NEWS` 310,906 (VisualNews — guardian 120,365 / bbc 71,947 / usa_today
62,441 / washington_post 56,153) and `COCO` 218,151 [MEASURED]. Manipulated images are made
by segmenting an object *in that same image* and replacing / removing / modifying it with
diffusion.

**Built-in OOD axis:** `distribution ∈ {ID: 289,948, OOD: 239,109}` and
`training ∈ {train: 86,542, val: 12,198, test: 430,317}` [MEASURED]. This is the only
candidate that ships an explicit ID/OOD label column.

**Their finding that matters here** [FETCHED]: "a model trained on a single image domain may
incorrectly learn that any distribution shift is evidence of manipulation, resulting in false
positives on images from a new domain." MMFusion trained on AUDITS-News and tested on
AUDITS-COCO — *same manipulation types* — dropped 8 AUC points.

**Partial download: excellent.** `train.zip` alone is 10.49 GB; train + val = 11.96 GB. The
whole thing fits the 20-40 GB budget with room to spare.

Note: the HF repo was created 2025-05-14 and last modified 2025-05-16, while the paper posted
2026-05-19. The upload predates the preprint by a year. I have not resolved that discrepancy
— **UNVERIFIED** whether the uploaded version is the final one described in the paper.

### 1.4 DailyBench — newest generators, and a fatal resolution confound

| Field | Value |
|---|---|
| Paper | *DailyBench: A Unified Benchmark for AI-Generated and Manipulated Images from Modern Generative Models* |
| Authors | Xin Jiang, Hao Tang, Junyao Gao, Meiqi Cao, Fei Shen, Dongming Zhang, Yongdong Zhang |
| arXiv | **2607.24016**, v1 2026-07-27, v3 2026-09-01 [FETCHED] |
| Venue | None stated — preprint |
| Size | Paper says "140K AI-generated and AI-manipulated images"; its own Table 1 row says 270K [FETCHED — the paper is internally inconsistent] |
| Disk | **20.2 GB** across 14 per-generator zips, plus a 2.95 GB `Qwen-Image_train.zip` at root [MEASURED, ModelScope file tree] |
| Classes | real / AI-generated (FakeBench) / manipulated (ManipulationBench) — your exact 3 classes |
| Masks | **No.** I unzipped `ManipulationBench/NanoBanana2.zip`: 611 entries, all images, only `0_real/` and `1_fake/` subtrees. No mask files. [MEASURED] |
| Generators | FakeBench: SD3.5, FLUX.1, FLUX.2, Z-Image, Qwen-Image, Nano Banana 2, GPT-Image 2. ManipulationBench: FLUX-Fill (random + object masks), FLUX.2-klein, Qwen-Edit, Step1X-Edit, GPT-Image2, Nano Banana 2 [FETCHED, project page] |

**Download:** the HuggingFace repo advertised on the project page and in the README,
`WhiteJiangzz/DailyBench`, **returns HTTP 401 — it does not exist publicly**, and an HF
dataset search for "DailyBench" returns an empty list [FETCHED]. The **ModelScope** mirror
`WhiteJiang/DailyBench` is live: API 200, Apache-2.0, 2,606 downloads [FETCHED]. ModelScope is
a Chinese host; expect slow transfers from India (I measured roughly 1-2 MB/s pulling the
small zips).

**Confound: catastrophic, and measured.** See §4.2. Real images are all 512×512, fakes all
1024×1024, so **resolution alone gives 100% held-out accuracy on both subsets I tested**
against a 58.6% baseline [MEASURED].
The paper's own Table 1 states this in the open — real AR 512×512, fake AR 777×774 — and
never discusses it.

The design intent was actually good: reals come from LAION-Aesthetics V2, and captions of
*those same reals* (via Qwen3-VL-8B) drive both the synthesis and the manipulation. The
provenance is matched. The output resolution was simply not controlled.

### 1.5 TGIF2 — matched provenance, but only 3,124 unique real images

| Field | Value |
|---|---|
| Paper | *TGIF2: Extended Text-Guided Inpainting Forgery Dataset & Benchmark* |
| Authors | Hannes Mareen, Dimitrios Karageorgiou, Paschalis Giakoumoglou, Peter Lambert, Symeon Papadopoulos, Glenn Van Wallendael |
| arXiv | **2603.28613**, 2026-03-30 [FETCHED] |
| Venue | Springer *Journal on Information Security*, DOI 10.1186/s13635-026-00235-9 [FETCHED] |
| Size | **271,788 manipulated images** derived from **3,124 authentic MS-COCO val2017 images**, 19 subsets [FETCHED] |
| Disk | **UNVERIFIED.** No size in the paper or README; WebDAV PROPFIND on the share links returns `NotAuthenticated`. ~272K images at 512-1024px is very likely well over 50 GB |
| Classes | pristine / spliced (SP) / fully-regenerated (FR) — a manipulation taxonomy, not real-vs-synthetic |
| Masks | **Yes** — semantic segmentation, bounding-box, and random non-semantic rectangular masks |
| Generators | Adobe Firefly (Photoshop 25.4.0), SD2, SDXL, FLUX.1 schnell / dev / Fill dev |
| Download | `github.com/IDLabMedia/tgif-dataset`, three Nextcloud shares on `cloud.ilabt.imec.be` (all HTTP 200), CC BY-SA 4.0 [FETCHED] |

**Same source pool: perfectly.** Every one of the 271,788 forgeries is an edit of one of the
3,124 COCO originals. That is the cleanest provenance matching in this survey — but with only
3,124 distinct real images it is a *localisation* benchmark, not a source of a real class you
could train a classifier on.

The SP/FR distinction is worth knowing about: "spliced" puts the edited region back into the
original image, "fully regenerated" passes the whole image through the diffusion process. FR
is the case INP-X shows detectors are actually solving.

### 1.6 Others found and checked

| Dataset | arXiv | Verdict |
|---|---|---|
| **Impostor** — *An Agent-Curated Benchmark for Realistic AIGC Manipulation Localization* | 2606.04545, 2026-06-03 | 100K manipulated + 100K real, all from **LVIS** — same pool, and they assign "each source image along with all associated manipulated images to the same subset to prevent data leakage" [FETCHED]. 7 editors incl. Nano Banana Pro, FLUX.1 Kontext, Qwen-Image-Edit. **No download link anywhere in the paper.** Worth an email to the authors |
| **XPlainVerse** | 2607.03562, 2026-07-03 | 1M images (470K real / 530K fake), HF `Abhijeet8901/XPlainVerse` exists but **`gated: "manual"`** [FETCHED]. ~287 GB; 2 GB tar shards so a 40 GB slice is easy, and `val/` alone is 39.63 GB. **No masks.** 76% of fakes are from one generator (Gemini-2.0-flash). Restrictive EULA |
| **MLLMGenSet** (GPT Image2 / Nano Banana2) | 2608.01258, 2026-08-02 | HF `zr-zhang/MLLM-Generated-Image-Detection-Dataset`, **ungated**, 3.32 GB [FETCHED]. Only **2,904 images** — an eval set, not training data. No masks. Strong on documents/receipts/screenshots. The paper **never states where its 726 real images came from** |
| **RealHD** | 2602.10546 (ACM MM 2025) | 730K images, inpainting masks included [FETCHED, abstract]. Project page `real-hd.github.io` and `github.com/Hanzhe-yu/RealHD` both resolve, but I could not extract download links or sizes from either. **UNVERIFIED** availability |
| **PromptForge-350k** | 2603.29386 | 354,258 edited pairs, masks. *"will be made publicly available"* — **no release** |
| **EditSleuth** | 2605.08695 | 257,725 triplets. *"We release the dataset..."* — **no URL given anywhere**, not on any author's HF namespace |
| **LocateEdit-Bench** | 2602.05577 | 231K edited images + masks. *"Dataset will be open-sourced upon acceptance"* — **no release** |
| **SIGMA** | 2605.27924 | ~1.1M mask annotations over AnyEdit / CrispEdit-2M / OmniEdit / PromptfixData. *"We'll release the full codebase as soon as the paper is accepted"* — **no release**. It is an annotator, not a corpus |
| **AIForge-Doc** | 2602.20569 | Documents/receipts only. Out of scope |
| **AGIDefect-4K**, **TextFake**, **TextRich**, **SurFITR**, satellite benchmark (2608.04840) | various 2026 | Narrow domains. Not pursued |

---

## 2. Availability check

All checked 2026-09-08. HF status is from the API (`200` = exists, `401` = not public).

| Dataset | Repo / host | Status | Gated | Files | Total size |
|---|---|---|---|---|---|
| NTIRE 2026 train | HF `deepfakesMSU/NTIRE-RobustAIGenDetection-train` | 200 | No | 6 shards | **114.36 GB** |
| NTIRE 2026 val | HF `deepfakesMSU/NTIRE-RobustAIGenDetection-val` | 200 | No | 2 zips + 2 label CSVs | **4.00 GB** |
| NTIRE 2026 test-public | HF `deepfakesMSU/NTIRE-RobustAIGenDetection-test-public` | 200 | No | zip + label CSV | **0.85 GB** |
| AUDITS | HF `DivyaApp/AUDITS` | 200 | No | 6 | **37.2 GB** |
| INP-X | Kaggle `emirhanbilgic/inpainting-exchange` | 200 | No | — | **10.73 GB** |
| OpenSDI train | HF `nebula/OpenSDI_train` | 200 | No | 70 parquet | **34.8 GB** (200,824 rows) |
| OpenSDI test | HF `nebula/OpenSDI_test` | 200 | No | 52 parquet, 5 splits | **18.3 GB** |
| So-Fake-Set | HF `saberzl/So-Fake-Set` | 200 | No | 2,343 parquet | **1,281.5 GB** |
| So-Fake-OOD | HF `saberzl/So-Fake-OOD` | 200 | No | 53 | **135.0 GB** |
| SID_Set | HF `saberzl/SID_Set` | 200 | No | 286 | **140.1 GB** |
| DailyBench | ModelScope `WhiteJiang/DailyBench` | 200 | No | 14 zips + 1 | **23.2 GB** |
| DailyBench (HF, as advertised) | HF `WhiteJiangzz/DailyBench` | **401** | — | — | **does not exist** |
| AIGIBench | HF `HorizonTEL/AIGIBench` | 200 | No | 53 zips | **236.63 GB** |
| OpenFake | HF `ComplexDataLab/OpenFake` | 200 | No | 645 | **3,440.75 GB** (2,493,222 rows) |
| OpenFakeTiny | HF `ComplexDataLab/OpenFakeTiny` | 200 | No | 8 | **4.88 GB** |
| DFBench | HF `IntMeGroup/DFBench` | 200 | No | 29 | **121.3 GB** |
| Community Forensics | HF `OwensLab/CommunityForensics` | 200 | No | 284 | **1,082.3 GB** |
| XPlainVerse | HF `Abhijeet8901/XPlainVerse` | 200 | **Yes, manual** | 156 | ~287 GB |
| MLLMGenSet | HF `zr-zhang/MLLM-Generated-Image-Detection-Dataset` | 200 | No | 4,358 | **3.32 GB** |
| TGIF2 | `cloud.ilabt.imec.be` × 3 shares | 200 | No | — | **UNVERIFIED** |
| PromptForge / EditSleuth / LocateEdit-Bench / SIGMA / Impostor | — | **no artifact** | — | — | — |

Two corrections to numbers circulating in our notes:

- **AIGIBench is not 288K images.** The string "288" does not appear in the arXiv HTML at all
  [FETCHED]. The paper states 72K (Setting-I) / 144K (Setting-II) train, and its test-subset
  prose sums to roughly 204K, with the abstract saying 23 subsets, the README 25, and 25 zips
  actually present. The GitHub README carries its own erratum correcting several test counts.
  Cite it as "UNVERIFIED, the source material is internally inconsistent".
- **OpenFake is not 4M downloadable images.** The v2 release is 2,493,222 rows / 3.44 TB and
  the frozen `v1.0` tag is 1,930,342 rows / 1,057 GB [FETCHED, datasets-server + HF API]. The
  "4M" traces to the v1 abstract's "three million real images paired with descriptive
  captions and almost one million synthetic counterparts" — captions, not shipped images.

---

## 3. Partial-download feasibility (budget: 20-40 GB)

| Dataset | Can you take a usable slice? | What 20-40 GB buys |
|---|---|---|
| **NTIRE 2026** | **Yes, officially blessed.** README: "All shards have similar data distribution, and can be used separately" | `shard_0.zip` = 20.59 GB ≈ 50K labelled images ≈ 18% of train. `shard_0+shard_5` = 31.96 GB ≈ 77K. Plus val (4.00 GB) + test-public (0.85 GB) with labels |
| **AUDITS** | **Yes — take all of it.** | `train.zip` 10.49 GB + `val.zip` 1.47 GB = 11.96 GB for the full train/val. Adding `test.zip` (25.18 GB) reaches 37.2 GB = 100% |
| **INP-X** | **Yes — take all of it.** 10.73 GB | 100% |
| **So-Fake-Set** | **Yes, and better than expected.** Shards are **shuffled** — one 0.5 GB shard already holds all 3 classes, masks and 20+ generators [MEASURED] | 40-70 shards ≈ 21-38 GB ≈ **1.7-3.0%** ≈ 34K-60K images, class-balanced and mask-bearing |
| **OpenSDI** | Yes, but **shards are class-ordered, not shuffled** — shard 0 is 100% `partial/fake` [MEASURED]. You must scan the `key` column to find class boundaries before choosing shards | 40 of 70 train shards ≈ 20 GB ≈ 57% of train, *if* you pick shards spanning all classes. Test is per-generator: flux 3.50, sd2 3.42, sd3 3.75, sdxl 3.66, sd15 4.10 GB |
| **DailyBench** | Yes — per-generator zips, 0.06 to 5.05 GB | The **whole thing is 23.2 GB**. Take all of it |
| **SID_Set** | Yes — 286 files / 140.1 GB | ~25% |
| **So-Fake-OOD** | Yes — 53 files / 135.0 GB | ~25% |
| **AIGIBench** | Yes, per-zip | `test/SDXL` 9.02 + `test/SD3` 9.98 + `test/FLUX1-dev` 8.78 + `test/DALLE-3` 8.96 = 36.7 GB. Or `val/` (4.38 GB) + `train/{car,cat,chair,horse}` (14.8 GB) = 19.2 GB |
| **OpenFake** | Yes | `reddit/test` config = 24.56 GB / 36,240 rows, self-contained. Or `OpenFakeTiny` at 4.88 GB |
| **XPlainVerse** | Yes, 2 GB tar shards; `val/` alone is 39.63 GB | Needs manual gate approval first |
| **Community Forensics** | Only via streaming or the 51.8K "PublicEval" subset | — |
| **TGIF2** | Per-model subdirectories, each with train/val/test | Fractions **UNVERIFIED** — no sizes published |

---

## 4. Confound check

This is the section that matters. Your finding was that file metadata (format + resolution)
predicts the class at 87.4%, above your model's 89%. I ran the same test on the candidates I
could get pixels for.

### 4.1 Method

For each dataset I downloaded real image bytes and read resolution, container format, colour
mode, file size and (for JPEG) the quantisation-table sum with PIL. I then fit a majority-vote
lookup table on a random half of the images and scored it on the held-out half, backing off to
the majority class for unseen keys. The held-out split matters: on datasets with many distinct
resolutions, fitting and scoring on the same images gives 90%+ purely by memorising
near-unique keys. Every number below is held-out.

This is packaged as **`scripts/data/metadata_confound.py`** — run it on anything you download
before you train on it:

```bash
venv-linux/bin/python scripts/data/metadata_confound.py path/to/dir_or_archive.zip
venv-linux/bin/python scripts/data/metadata_confound.py shard.parquet --key-col key
```

It reads directories (class = first subdirectory, so `0_real/1_fake` works), zip archives and
parquet shards, and prints each metadata feature's held-out accuracy with its margin over the
majority baseline, flagging anything more than 10 points clear as a leak.

### 4.2 Results [MEASURED]

Best metadata feature per dataset, held-out accuracy vs majority-class baseline:

| Dataset | Sample | Best feature | Held-out acc | Baseline | Margin | Verdict |
|---|---|---|---|---|---|---|
| **DailyBench** (NanoBanana2, both subsets) | 1,101 | resolution | **100.0%** | 58.6% | **+41.4** | **Fatal** |
| **So-Fake-Set** shard 0, 3-class | 850 | megapixels | **71.1%** | 43.5% | **+27.6** | **Serious** |
| **NTIRE 2026** val | 336 | aspect ratio | **60.1%** | 53.0% | +7.1 | **Clean** (within ~2σ) |
| **OpenSDI** train shard 0 | 2,869 | — | single class in shard | — | — | Format-clean; see below |

Full breakdown:

| Feature | DailyBench | So-Fake-Set (3-class) | NTIRE val |
|---|---|---|---|
| baseline | 58.6% | 43.5% | 53.0% |
| container format | 62.1% | 56.2% | 47.0% |
| colour mode | 62.1% | 44.0% | — |
| resolution (exact) | **100.0%** | 68.2% | 56.5% |
| megapixels (0.1 bucket) | **100.0%** | **71.1%** | 58.9% |
| aspect ratio (0.1 bucket) | 62.1% | 63.8% | **60.1%** |
| file size (10 KB bucket) | **90.2%** | 55.5% | — |
| JPEG quantisation-table sum | 62.1% | 54.6% | 47.0% |
| all combined | 99.3% | 58.4% | 56.5% |
| *baseline* | *62.1%* | *43.5%* | *53.0%* |

(DailyBench's baseline differs between the two tables because §4.2's first table pools its two
subsets and this one reports the ManipulationBench subset, on which the tool was run.)

Read the NTIRE column carefully: **container format and JPEG quantisation table carry
literally zero information** — every image in both classes is JPEG with an identical
quantisation table (sum 369). Reals span 141 distinct resolutions across 166 images and fakes
131 across 170, with heavily overlapping medians (640,000 vs 562,176 pixels). The 60.1%
aspect-ratio figure is +7.1 points on a 168-image held-out set, roughly 1.9 standard errors —
suggestive at most. This is what a controlled dataset looks like.

**DailyBench, in full detail.** Real images: 228/228 JPEG, 228/228 at 512×512, quantisation
table sum 1858 for every one. Fake images: 383/383 JPEG, 382 at 1024×1024 and 1 at 1408×768,
same qtable sum 1858. So the JPEG re-encoding *was* uniformised — and then the resolution was
left as a perfect giveaway. The FakeBench subset repeats it exactly: reals 245/245 at
512×512, fakes 235/245 at 1024×1024 and 10 at 1408×768.

If you use DailyBench at all, you must resize or random-crop both classes to a common size
*before* the model sees them, and you must say in the write-up that you did, because a number
reported on raw DailyBench is not measuring detection.

**So-Fake-Set, in full detail.** From one shuffled shard (850 images: 369 full_synthetic,
256 tampered, 225 real):

| Class | JPEG | PNG | median pixels | median bytes | masks |
|---|---|---|---|---|---|
| real | 151 | 74 | 187,500 | 103,912 | 0 / 225 |
| full_synthetic | 36 | 333 | 589,824 | 865,963 | 0 / 369 |
| tampered | 34 | 222 | 262,144 | 366,161 | **256 / 256** |

Format alone: JPEG is 68% real, PNG is 60% synthetic-or-tampered. Resolution is worse — reals
cluster at 178×218 (CelebA-style) and 256×256, full_synthetic at 1024×1024, tampered at
exactly 512×512 or 1024×1024. **This is your bug, in the dataset you were about to train on.**
71.1% held-out metadata-only against a 43.5% baseline is lower than your 87.4% but the same
failure.

Mitigation is the same as for your own corpus: re-encode every image to a single container at
a single quality, and resize/crop to a fixed size, before any split is drawn.

**OpenSDI.** Sampling the `key` column across offsets, every one of the four categories
(`entire/real`, `entire/fake`, `partial/real`, `partial/fake`) is `.jpg` in the train split
[MEASURED]. Shard 0 (2,869 rows, all `partial/fake`) is 100% JPEG with a wide spread of
quantisation tables (many distinct qtable sums, not one constant) and resolutions dominated by
1024×683 and 1024×768 — i.e. native Megalith photo geometry, which the local edits preserve by
construction. One leak found: the `sd15` **test** split contains a small group of bare-filename
`.png` fakes alongside `.jpg` reals. Drop those or re-encode.

### 4.3 What the papers say about sourcing

| Dataset | Reals from | Fakes from | Shared pool? |
|---|---|---|---|
| **NTIRE 2026** | CC12M, CommonPool, RedCaps | captions of *those reals* → 42 generators, then resolution / aspect / JPEG-QF distributions aligned to the reals | **Yes, and explicitly bias-controlled** |
| **OpenSDI** | Megalith-10M ("wholesome, unedited, copyright-free licensed images") | "Based on an authentic set of images from Megalith-10M... our pipeline utilizes VLMs to produce varied manipulation instructions" | **Yes** |
| **AUDITS** | VisualNews + MS COCO | same images, objects segmented then replaced/removed/modified | **Yes** |
| **INP-X** | CelebA-HQ, CityScapes, OpenImages, SUN-RGBD | inpainted versions of those same images | **Yes** |
| **TGIF2** | 3,124 MS-COCO val2017 | those same 3,124, inpainted 6 ways | **Yes** |
| **Impostor** | LVIS | those same images, object-level edits | **Yes** |
| **DailyBench** | LAION-Aesthetics V2 (aesthetic ≥6, shortest side ≥512) | captions of those reals (Qwen3-VL-8B) → synthesis and manipulation | Yes in provenance — **but broken by resolution** |
| **So-Fake-Set** | mixed (OpenForensics, CelebA-scale faces, COCO-scale scenes) | 35 generators | Partly — **and it leaks via format and resolution** |
| **XPlainVerse** | OpenImages V7, EMOTIC, PIPA, PIC 2.0, PISC | edits *of those same reals* | Yes |
| **AIGIBench** | FFHQ, CelebA-HQ, Open Images V7 — "randomly selected and merged an equal number of images from these sources... matching the quantity of fake images" | separate generator outputs | **No — count-matched only.** Same hazard as GenImage |
| **OpenFake** | LAION-filtered, Pexels, DOCCI, ImageNet, Reddit — *deliberately disjoint* between train and test | shared prompt bank derived from captioned reals | Prompt-level only; pixel pools intentionally disjoint |
| **GenImage** (background) | ImageNet JPEGs | generator PNGs | **No — this is the canonical instance of the bug** |

Flagged as likely carrying the one-corpus-per-class problem: **GenImage**, **AIGIBench**, and
any of the "20-corpus union" constructions including your own.

---

## 5. Live benchmarks accepting submissions

Fully covered in `docs/BENCHMARK_LIVENESS_2026-09-08.md`, verified the same day. Summary:

- **Nothing image-based is open.** The NTIRE 2026 challenge (Codabench 12761) closed its
  Testing Phase 2 on **2026-09-05**, three days ago — all four phases now read `Previous`
  [FETCHED]. 541 participants, 3,449 submissions, scored by ROC-AUC.
- **But its data and labels are now public**, including `val_labels.csv` (10,000 rows) and
  `test_labels.csv` (2,500 rows) [FETCHED]. So you can compute the challenge's own metric
  offline and quote it against the published leaderboard, which is `hidden: false`. That is
  the closest thing to an externally comparable number available today, and it costs nothing.
- The only live deepfake competition anywhere is **RTC-SDD** (Codabench 17381, phase
  `Current` to 2026-11-09) and it is **speech/audio**.
- **FaceForensics++** infrastructure is up but its leaderboard has been frozen at 109 entries
  since 2022-10-06.
- **Papers With Code is gone** — `paperswithcode.com` now redirects to
  `huggingface.co/papers/trending`.
- Diary note: NTIRE opened mid-January 2026. Set a reminder for **early January 2027**.

---

## 6. Ranked recommendation

Budget 20-40 GB, one RTX 4060 8 GB, WSL capped at 8 GB RAM, 3 classes + masks.

### Primary training set: **OpenSDI train** (34.8 GB, or ~20 GB in shards)

`nebula/OpenSDI_train`, CC-BY-SA-4.0, ungated, 200,824 rows.

It is the only downloadable dataset that gives you **all three of your classes, with masks,
from a single real-image pool, in a single format**:

- The `key` field is prefixed `entire/` (fully synthetic) or `partial/` (locally edited) —
  so `real / entire / partial` maps straight onto real / AI-generated / AI-edited. Deriving
  the 3-class label is one string split. [MEASURED]
- Masks ship on the `partial/fake` rows exactly. [MEASURED]
- Reals and fakes both descend from Megalith-10M. Every train-split category is `.jpg`.
- Parquet + `datasets` streaming means you never hold it in 8 GB of RAM.

**One thing I could not measure.** OpenSDI's shards are class-ordered, so the one shard I
downloaded (2,869 rows) was 100% `partial/fake` — which means I have **not** run the held-out
metadata test across its three classes. What I do know is that every train-split category is
`.jpg`, that the quantisation tables vary widely rather than sitting at one constant, and that
`partial` edits preserve the original photo's geometry by construction. The `entire` class is
the one to check: a full regeneration need not keep the source resolution, and that is exactly
where DailyBench broke. **Run `scripts/data/metadata_confound.py` on a class-spanning shard
selection before you train.**

Two other things to handle. The `entire` class is the minority — my offset sampling suggests roughly
1 in 10 of the fakes, so on the order of 10K fully-synthetic images against ~90K edited
(**approximate, from a small sample — count it properly before you rely on it**). And the
shards are class-ordered, so scan the `key` column across all 70 shards first and pick a
spanning subset rather than taking shards 0-39.

### Held-out / OOD benchmark: **NTIRE 2026 val + test-public** (4.85 GB)

`deepfakesMSU/NTIRE-RobustAIGenDetection-val` and `-test-public`. 12,500 labelled images
across three difficulty tiers, 42 generators, 36 transformations, ungated, labels public.

This is the strongest external reference available: it is the only dataset in the survey whose
authors deliberately aligned resolution, aspect ratio and JPEG quality between the classes, so
a number measured on it cannot be explained away as metadata leakage. Collapse your three
classes to binary to score it, and report the collapse rule you used. It costs under 5 GB and
gives you a figure you can put next to a published CVPR-workshop leaderboard.

Add `shard_0.zip` (20.59 GB) later if you want to fine-tune on it.

### Third pick, if you want the confound result itself: **INP-X** (10.73 GB)

Run your best checkpoint on `originals` / `standard_inpainting` / `inpainting_exchange` and
report the three accuracies. If your accuracy collapses from the standard-inpainting number to
the exchange number, you have reproduced the paper's 91%→55% finding on your own model — which
is a far more interesting chapter than another in-domain accuracy, and it is the same story
your generalisation results already tell. It is 10.7 GB, MIT-licensed, and ships masks.

### Total: ~50 GB, or ~36 GB if you take OpenSDI as 20 GB of shards.

### What to drop

- **So-Fake-Set as a primary training set.** Not because of size — the shuffled shards fix
  that — but because a 3-class metadata lookup gets 71.1% held-out on it against a 43.5%
  baseline. If you use it, re-encode and
  resize first, and report the metadata-only baseline alongside your model's accuracy, exactly
  as you did for your own corpus.
- **DailyBench as a benchmark.** 100% held-out metadata separability. It is still the best source of
  *recent* generators (FLUX.2, GPT-Image 2, Nano Banana 2, Z-Image, Qwen-Image), so keep it as
  a qualitative probe after forcing both classes through identical resize + re-encode — but do
  not put a raw DailyBench accuracy in the thesis.
- **AIGIBench.** 236 GB, count-matched reals from a different pool, and its own counts do not
  reconcile.
- Anything in §1.6 marked "no release".

### Sanity check to run before training on any of them

Run your existing metadata-only classifier — format, resolution, file size, JPEG quantisation
table — on whatever you download, *before* you train. It takes minutes and it is the number
that told you your own dataset was broken. Report it next to every accuracy you publish.
