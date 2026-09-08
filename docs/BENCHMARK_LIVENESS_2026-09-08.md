# Which deepfake / AI-image detection benchmarks are actually live

Verified 2026-09-08. Every claim below is tagged either **[FETCHED]** (I pulled the page
or API myself and am quoting it) or **[SNIPPET]** (a search summary I could not confirm by
fetching). Anything I could not establish is marked **COULD NOT VERIFY**.

Ranked by usefulness to a solo student with one laptop-class GPU who needs an
externally-comparable number.

---

## Headline findings

1. **FaceForensics++ is technically live but community-dead.** The server, the
   registration form, the 565 MB benchmark download and the documented submission format
   all work today. But the public leaderboard has been frozen at **exactly 109 entries
   since ~2022** — four years without a single new published entry.
2. **A sibling benchmark on the same server is still growing**, which proves the
   evaluation infrastructure itself is alive. See the diagnostic below — it is the single
   most useful thing in this report.
3. **No image/video deepfake-detection competition is open for submission today.** The
   closest one — NTIRE 2026 "Robust AI-Generated Image Detection in the Wild" at CVPR —
   **closed three days ago, on 2026-09-05.**
4. **Papers With Code is gone.** `paperswithcode.com` now 302-redirects to
   `huggingface.co/papers/trending` **[FETCHED]**. The old route to "look up the SOTA table
   and compare" no longer exists.
5. The best realistic option for an external number this year is **Deepfake-Eval-2024** —
   a gated but active, small (20.3 GB, 1,975 images), evaluation-only in-the-wild
   benchmark with published baselines you can compare against directly.

---

## 1. FaceForensics++ benchmark — **LIVE, but the board is frozen**

**URL:** https://kaldir.vc.in.tum.de/faceforensics_benchmark/

### (a) Is it live?

**Yes, the infrastructure is up.** All fetched today:

| Thing | Status |
|---|---|
| Leaderboard page | **[FETCHED]** HTTP 200, renders the full table |
| `/login` | **[FETCHED]** HTTP 200, live login form |
| `/register` | **[FETCHED]** HTTP 200, live server-rendered registration form with real input fields (`firstname`, `lastname`, `email`, `website`, `username`, `password`) |
| `/documentation` | **[FETCHED]** HTTP 200, full submission policy and format |
| `faceforensics_benchmark_images.zip` | **[FETCHED]** HTTP 200, `Content-Length: 592663962` (565 MB), `Last-Modified: Tue, 24 Sep 2019` |
| `faceforensics_benchmark_exsample_submission.zip` | **[FETCHED]** downloaded, unzipped, inspected |

### (b) The board has not moved in four years

There are **no dates on the leaderboard** — I checked, the only column controls are sort
links (`sortby=ddeepfakes`, `dface2face`, `dfaceswap`, `dneuraltextures`, `dpristine`) and
there is **no pagination**. So I dated it by counting table cells across Wayback snapshots
**[FETCHED, via CDX + direct snapshot fetches]**:

| Snapshot | Entries |
|---|---|
| 2019-05-02 | 5 |
| 2019-11-15 | 9 |
| 2020-06-30 | 44 |
| 2020-12-31 | 71 |
| 2021-08-01 | 99 |
| 2021-12-09 | 106 |
| 2022-10-06 | **109** |
| 2024-12-01 | 109 |
| 2025-10-03 | 109 |
| 2026-02-19 | 109 |
| **live today** | **109** |

The Wayback CDX content digest is *identical* across every snapshot from 2022-10-06 to
2026-02-19 and matches the live page. I fetched the 2022-10-06 snapshot directly and its
top-row scores are byte-for-byte the ones on the board right now.

Corroborating: the newest paper cited anywhere on the board is **arXiv 2004.11804** (2020)
and a Springer article from 2021 **[FETCHED from the live page's link list]**.

**So: the most recent entry you can see dates to roughly late 2021 / 2022. Not 2024, as
`docs/EXTERNAL_BENCHMARK.md` currently guesses.**

### (c) The diagnostic that matters

The FF++ benchmark runs on the same server, same codebase and same lab as the **ScanNet
benchmark** (the About page **[FETCHED]** literally thanks the ScanNet authors for the
website sources). ScanNet's board **[FETCHED]**:

| Snapshot | Cells |
|---|---|
| 2025-12-16 | 9,187 |
| 2026-02-07 | 9,583 |
| 2026-04-22 | 9,583 |
| **live today** | **10,453** |

ScanNet gained ~1,270 cells in the last nine months. **The TUM evaluation infrastructure is
demonstrably still scoring and publishing submissions in 2026.** FF++ is frozen because
nobody submits to it, not because the server died.

**Honest caveat:** I cannot prove the FF++ scorer still returns a score, because that would
require registering and burning a submission. It is possible the FF++-specific scoring
backend is broken while ScanNet's works. What I can say is that everything up to the point
of submission works, and the general infrastructure is alive.

### (d) Submission format — **verified, including your rate-limit belief**

**[FETCHED verbatim from `/documentation`]:**

> Results for a method must be uploaded as a single .zip or .7z file, which when unzipped
> must contain a single .json file containing a dictionary of your predicted labels for all
> benchmark images. With other words: There must not be any additional files or folders in
> the archive except those specified below. For each image filename the dictionary in the
> json file should contain one of the two labels "fake" and "real".

I downloaded and unzipped the official example. It is one file, `example_submission.json`,
24,002 bytes, **1,000 entries**, of exactly this shape:

```json
{
    "0000.png": "fake",
    "0001.png": "fake",
    "0002.png": "real",
    ...
}
```

**Your 2-week rate limit belief is CORRECT** — here is the exact wording **[FETCHED,
appears on both `/login` and `/documentation`]**:

> To help enforcing this policy, we block updates to the test set results of a method for
> two weeks after a test set submission.

Note the precise semantics: it blocks *updates to the results of a method* for two weeks.
Also **[FETCHED]**:

> Evaluating on the test data via this evaluation server must only be done once for the
> final system.

and

> It is not permitted to register on this webpage with multiple e-mail addresses. We will
> ban users or domains if required.

### (e) Top of the board (live today, **[FETCHED]**)

| Method | Deepfakes | Face2Face | FaceSwap | NeuralTextures | Pristine | Total |
|---|---|---|---|---|---|---|
| DirechletEnsemble-Classifier | 0.955 | 0.956 | 0.932 | 0.967 | 0.992 | **0.973** |
| Beijing ZKJ | 1.000 | 0.920 | 0.961 | 0.880 | 0.948 | 0.941 |
| ZAntiFakeBio | 1.000 | 0.920 | 0.971 | 0.907 | 0.936 | 0.940 |
| Leo | 1.000 | 0.861 | 0.971 | 0.853 | 0.922 | 0.917 |
| MixingExpert | 1.000 | 0.905 | 0.942 | 0.780 | 0.922 | 0.909 |
| NoSenseAtAll | 0.982 | 0.905 | 0.951 | 0.827 | 0.908 | 0.908 |
| Cancer | 0.964 | 0.781 | 0.942 | 0.780 | 0.952 | 0.903 |
| RobustForensics | 0.991 | 0.891 | 0.951 | 0.807 | 0.904 | 0.902 |
| ENbsoftFW1111 | 0.982 | 0.891 | 0.922 | 0.787 | 0.898 | 0.892 |
| Aquarius | 1.000 | 0.854 | 0.971 | 0.807 | 0.884 | 0.890 |

The 0.973 figure in `docs/EXTERNAL_BENCHMARK.md` is correct.

### (f) The GitHub repo is unmaintained

**[FETCHED from the GitHub API]** `ondyari/FaceForensics`:
- `pushed_at`: **2022-12-08** (last code change, ~3¾ years ago)
- `updated_at`: 2026-09-08 (this is just metadata/stars, not code)
- `open_issues_count`: **96**
- `archived`: **false**
- stars: 2,772

**No retirement or freeze notice exists** in the README **[FETCHED]** — the benchmark
section reads as normal. But the issue tracker tells the real story **[FETCHED]**: the 12
most recent issues are all **open with no maintainer responses**, including
**"kaldir.vc.in.tum.de server is down"** (opened 2025-10-05, still open) and repeated
dataset-download failures. Recent issues in 2026-03 and 2025-12 are people asking for
dataset access via the tracker, which suggests the Google access form may not be answered
promptly.

Dataset access is still via a Google form **[FETCHED from the README and `/documentation`,
same form link on both]**:
`https://docs.google.com/forms/d/e/1FAIpQLSdRRR3L5zAv6tQ_CKxmK4W96tAab_pfBu2EKAgQbeDVhmXagg/viewform`
— the README warns: *"If you have not received a response within a week, it is likely that
your email is bouncing."* **You do not need this form for the benchmark itself** — the
benchmark images zip is a direct, unauthenticated download.

### (g) Can a solo student with one GPU do this?

**Yes, trivially, and it is the cheapest external number available to you.** 1,000 images,
565 MB, inference only. No training required, no access request required for the benchmark
set. Registration is a plain web form.

**The risk is not compute, it is whether your entry ever appears.** Four years of zero new
entries means you may submit and get nothing back. Budget it as an hour of work with an
uncertain payoff, not as the centrepiece of your evaluation chapter.

---

## 2. Deepfake-Eval-2024 — **LIVE and actively maintained. Best real option.**

**URL:** https://huggingface.co/datasets/nuriachandra/Deepfake-Eval-2024
**Paper:** arXiv **2503.02857**, *"Deepfake-Eval-2024: A Multi-Modal In-the-Wild Benchmark
of Deepfakes Circulated in 2024"* **[FETCHED]**
**Authors [FETCHED]:** Nuria Alina Chandra, Hannah Lee, Ryan Murtfeldt, Lin Qiu, Arnab
Karmakar, Emmanuel Tanumihardja, Kevin Farhat, Ben Caffee, Changyeon Lee, Jongwook Choi,
Sejin Paik, Aerin Kim, Oren Etzioni
**Venue:** arXiv preprint, v1 2025-03-04, **latest v5 2026-05-27**. No conference
proceedings confirmed. **[FETCHED]**

### (a) Live?

**Yes, and recently touched.** HF API **[FETCHED]**: `lastModified` **2026-08-11**,
`downloads` 619, `likes` 46, `gated: "manual"`, `license: cc-by-sa-4.0`. Dataset card notes
a metadata correction applied 2025-10-29.

### (b) Task and label space

Binary real/fake, manually labelled, **in the wild** — media actually circulated in 2024,
88 website sources, 52 languages. **[FETCHED]** 44 hours video + 56.5 hours audio +
**1,975 images (767 real / 1,208 fake)**, mode resolution 1024×1024.

### (c) How you get a comparable number

There is **no eval server and no leaderboard**. But the paper publishes baselines on the
public test set, so you run your model and quote against their table **[FETCHED from the
paper HTML]**:

| Model | AUC on Deepfake-Eval-2024 | Acc | AUC on its original benchmark | Acc |
|---|---|---|---|---|
| UFD | 0.56 | 0.63 | 0.94 | 0.81 |
| DistilDIRE | 0.52 | 0.61 | 0.99 | 0.98 |
| NPR | 0.53 | 0.47 | 0.98 | 0.94 |

Average AUC drop for image models: **~45%** vs the academic datasets they were tested on.
Best commercial image detector: **accuracy 0.82, AUC 0.90**, precision 0.99, recall 0.71,
F1 0.83. Human forensic analysts estimated at ≥90% accuracy.

Finetuning on 60% of the data barely helps (UFD 0.56 AUC, DistilDIRE 0.56, NPR 0.55) —
which is itself a citable result.

### (d) Solo student with one GPU?

**Yes — this is the best fit in the whole report.** 20.3 GB total; the image subset is only
1,975 images. Inference-only.

**Two catches:**
1. **Gated with manual approval.** The form **[FETCHED verbatim]** demands *"Link to a
   verifiable source with evidence (ex website, paper, or news releases) that you have done
   work related to deepfake detection or a related field"*, and warns *"Incomplete answers
   or responses that fail to directly address the question will lead to rejection. There is
   no option to edit your answers later."* As an MSc student you would want to point at a
   supervisor page, a thesis registration, or this repo. Apply early — approval is manual.
2. **Evaluation only.** Terms **[FETCHED verbatim]**: *"Users may only use this dataset for
   evaluation. Use of this dataset for training goes against the terms of use."* That is
   fine for your purpose, but do not finetune on it.

---

## 3. GenImage — **dataset live, no leaderboard, self-reported numbers**

**Paper [FETCHED]:** *"GenImage: A Million-Scale Benchmark for Detecting AI-Generated
Image"*, arXiv **2306.08571**, v1 2023-06-14, v2 2023-06-24.
**Authors [FETCHED]:** Mingjian Zhu, Hanting Chen, Qiangyu Yan, Xudong Huang, Guanyu Lin,
Wei Li, Zhijun Tu, Hailin Hu, Jie Hu, Yunhe Wang.
**Venue:** NeurIPS 2023 Datasets & Benchmarks — note the arXiv page's Comments field does
**not** name the venue **[FETCHED]**; the D&B acceptance is well known but I did not fetch
a proceedings page to confirm it. Treat as **[SNIPPET]** for the venue specifically.

### (a) Live?

**Yes, downloadable.** **[FETCHED]**
- GitHub `GenImage-Dataset/GenImage`: `pushed_at` **2024-04-02**, `updated_at` 2026-08-30,
  580 stars, 2 open issues, not archived. **No dead-link or unavailability notice.**
- Official Google Drive folder: **HTTP 200** (uploaded 2024-03-15 per the README).
- Baidu Yunpan mirror, access code `ztf1`.
- Project page https://genimage-dataset.github.io/ loads, but still says *"This Website is
  still updating."* Its only download link is the Baidu one.
- Third-party HF mirrors exist and are ungated: `jzousz/GenImage` (10,822 downloads,
  lastModified 2024-12-30), `e8035669/GenImage` (2025-09-26).

### (b) Task, label space, protocol

Binary real/fake. **Your understanding of the protocol is correct** **[FETCHED]**: train on
one generator, test across all eight — **Midjourney, SD v1.4, SD v1.5, ADM, GLIDE, Wukong,
VQDM, BigGAN**. The conventional setup is *train on SD v1.4, test on all 8*. A second task
covers degraded images (low-res, blurred, JPEG-compressed).

### (c) Current SOTA

There is **no leaderboard** — not in the README, not on the project page. Numbers are
self-reported in papers. The reference point most people cite is the AIDE paper's table
**[FETCHED from arXiv 2406.19435 HTML]**, models trained on SD v1.4:

| Method | MJ | SD1.4 | SD1.5 | ADM | GLIDE | Wukong | VQDM | BigGAN | **Mean** |
|---|---|---|---|---|---|---|---|---|---|
| PatchCraft | 79.00 | 89.50 | 89.30 | 77.30 | 78.40 | 89.30 | 83.70 | 72.40 | **82.30** |
| AIDE | 79.38 | 99.74 | 99.76 | 78.54 | 91.82 | 98.65 | 80.26 | 66.89 | **86.88** |

### (d) Solo student with one GPU?

**Partially.** The full set is **~679 GB** **[FETCHED from the `jzousz/GenImage` HF card]**
— you are not downloading that. But the protocol only needs the SD v1.4 *train* subset plus
the eight *test* subsets, and the test subsets are small. Pull per-generator folders from
Drive rather than the whole archive. Training one detector on one generator's subset is
feasible on a laptop GPU; it is the standard student entry point.

**No external validation, though** — you compute your own number on a public test set. It
is comparable-by-convention, not adjudicated.

---

## 4. Chameleon — **hardest benchmark here, but email-gated and no eval server**

**Your recalled title was wrong.** Corrected **[FETCHED from arXiv]**:

- **Exact title:** *"A Sanity Check for AI-generated Image Detection"* — **not** "Are We on
  the Right Way for Evaluating AI-generated Image Detection?"
- **Authors:** Shilin Yan, Ouxiang Li, Jiayin Cai, Yanbin Hao, Xiaolong Jiang, Yao Hu,
  Weidi Xie
- **arXiv:** 2406.19435, v1 2024-06-27, v3 2025-02-15
- **Venue:** **ICLR 2025**. The arXiv Comments field does not state it **[FETCHED]**, but
  the GitHub repo header does and there is an ICLR 2025 proceedings PDF **[SNIPPET, from
  proceedings.iclr.cc]**.
- **Chameleon is the dataset; AIDE is the method.** Do not cite them as the same thing.

### (a) Live?

**Repo yes, dataset by request only.** GitHub `shilinyan99/AIDE` **[FETCHED API]**:
`pushed_at` **2025-06-04**, `updated_at` 2026-09-08, 332 stars, 11 open issues, not
archived. News entries: Chameleon released 2024-12-29, ICLR acceptance 2025-01-23.

**Access [FETCHED]:** *"please send an email to tattoo.ysl@gmail.com."* Academic use only,
commercial use prohibited, and the paper says users must sign an EULA with *"access
contingent upon thorough review and subsequent approval."* I searched the HuggingFace
datasets API for "Chameleon" and **none of the 25 results is this dataset** **[FETCHED]**.
There is **no evaluation server** — you evaluate locally with their scripts.

### (b) What it is, and how hard

~**26,000 test images** — 14,863 real, 11,170 AI-generated — across human, animal, object,
scene; 720p to 4K; every fake passed a human Turing test. Real images from Unsplash, fakes
scraped from ArtStation, Civitai and Liblib. Filtered to ≥448×448, deduplicated, NSFW- and
CLIP-filtered. **[FETCHED]**

**It is brutal.** Full table **[FETCHED from arXiv HTML, Table 5]**, mean accuracy and the
fake/real per-class split:

| Detector | Mean Acc | Fake acc / Real acc |
|---|---|---|
| CNNSpot | 56.94% | 0.08% / 99.67% |
| FreDect | 55.62% | 13.72% / 87.12% |
| Fusing | 56.98% | 0.01% / 99.79% |
| GramNet | 58.94% | 4.76% / 99.66% |
| LNP | 57.11% | 0.09% / 99.97% |
| UnivFD | 57.22% | 3.18% / 97.83% |
| DIRE | 58.19% | 3.25% / 99.48% |
| PatchCraft | 53.76% | 1.78% / 92.82% |
| NPR | 57.29% | 2.20% / 98.70% |
| **AIDE** (their own) | **56.45%** | 0.63% / 98.46% |

Read the second column, not the first. **Every detector, including the paper's own, detects
essentially zero fakes.** The ~57% "accuracy" is just the real-class majority. Even AIDE,
which beats SOTA on GenImage and AIGCDetectBenchmark, collapses to 0.63% on fakes here.

### (c) Solo student with one GPU?

Compute-wise yes — 26k images, inference only. **The blocker is the email gate**, with no
stated turnaround. Worth emailing now in parallel with everything else, since it costs
nothing.

---

## 5. WildFake — **public and huge; paper metadata is a trap**

**There are two different titles and two different author lists. Cite the AAAI one.**

- **arXiv [FETCHED]:** *"WildFake: A Large-scale **Challenging** Dataset for AI-Generated
  Images Detection"*, arXiv **2402.11843**, **v1 only, 2024-02-19**, authors listed as just
  **Yan Hong, Jianfu Zhang** (2 authors), no venue in Comments.
- **AAAI proceedings [FETCHED by subagent]:** *"WildFake: A Large-Scale **and Hierarchical**
  Dataset for AI-Generated Images Detection"*, **AAAI-25**, Vol. 39 No. 4, pp. 3500–3508,
  DOI `10.1609/aaai.v39i4.32363`. Authors: **Yan Hong, Jianming Feng, Haoxing Chen, Jun Lan,
  Huijia Zhu, Weiqiang Wang, Jianfu Zhang** (7 authors, Ant Group + SJTU).

The arXiv preprint was never updated to the camera-ready — different title, **five missing
co-authors**. If you cite the arXiv metadata you will get the authorship wrong.

### (a) Live?

**Yes, publicly downloadable — I verified this directly**, correcting an earlier
inconclusive result. ModelScope API for `hy2628982280/WildFake` **[FETCHED]**:
- `Visibility: 3` (public), `License: Apache License 2.0`
- **`Downloads`: 223,558**
- **`StorageSize`: 1,291,478,056,101 bytes ≈ 1.17 TB**
- Created 2025-02-08, `LastUpdatedTime` **2025-03-27**

Companion GitHub `hy-zpg/AIGC-Image-Detection-Dataset` **[FETCHED]** gives ModelScope as the
only download link — no HF, Drive or Baidu mirror, no size, no news entries, **no leaderboard
and no evaluation protocol described**.
Unofficial HF subsets exist: `techjam-aigc/wildfake-eval-subset` (110K rows),
`buxtcodes/WildFake-Sample` (30K rows) **[SNIPPET — found via HF search, not individually
fetched]**.

### (b) Task

Binary real/fake, but organised hierarchically for targeted generalisation tests:
cross-time (early vs latest GANs), cross-architecture (SD / DDPM / DALL-E 2), cross-weight
(base vs finetuned vs adapter SD), cross-version (DALL-E, Midjourney). 3.7M images.

### (c) Solo student with one GPU?

**Not the full thing — 1.17 TB.** Use one of the unofficial subsets, or pull single
hierarchy branches from ModelScope. Note ModelScope is a Chinese host; expect slow
transfers from India. No eval server, no leaderboard — self-reported numbers only.

---

## 6. Competitions and challenges — **nothing open today**

Every phase status below came from the Codabench public API (`status` is `Previous` /
`Current` / `Next` with real timestamps), which is the most reliable liveness signal
available. I re-verified the three most consequential entries myself.

### The near miss

**NTIRE 2026 Robust AI-Generated Image Detection in the Wild** — Codabench id 12761
**[FETCHED, I verified this directly]**:

| Phase | Window | Status |
|---|---|---|
| Intro | 2026-01-15 → 2026-01-30 | Previous |
| Validation | 2026-01-31 → 2026-03-09 | Previous |
| Testing | 2026-03-10 → 2026-03-15 | Previous |
| **Testing Phase 2** | **2026-08-30 → 2026-09-05** | **Previous** |

**It closed three days ago.** 541 participants, 3,449 submissions. Scored by ROC-AUC. This
is exactly the competition you wanted: CVPR-affiliated, image-only, AI-generated detection,
in the wild. The competition is still `published: true` with
`registration_auto_approve: true` **[FETCHED]**, so you can likely still register and read
the leaderboard even though submissions no longer score.

**Actionable:** it is annual and opened **mid-January 2026**. Put a reminder for early
January 2027.

### Everything else, all closed

| Challenge | Venue | Ends | Status |
|---|---|---|---|
| NTIRE 2026 Robust Deepfake Detection (id 12795) | CVPR 2026 | 2026-03-23 | **Closed** [FETCHED, verified] — 365 participants, 5,228 submissions; report at arXiv 2604.24163 |
| Explainable Deepfake Detection / XPlainVerse (id 16461) | ACM MM 2026 | 2026-06-18 | **Closed**. Successor to 1M-Deepfakes. 144 participants, 414 submissions. Image real/fake + two text explanations; 760K images; submit zip of 3 JSONL; ≤1/day, ≤10 total. Its official site `i-am-shreya.github.io/deepfake_explainability_mm_2026/` **404s** |
| 1M-Deepfakes Detection Challenge | ACM MM 2024, 2025 | 2025 | **Closed**. 2024: 191 teams, 1,034 submissions (arXiv 2409.06991). 2025 = AV-Deepfake1M++, ~2.1M clips, ACM MM 2025 Dublin |
| DDL-X: Deepfake Detection, Localization & Explainability (id 15686) | IJCAI 2026 | 2026-06-08 | **Closed** — organizer page literally shows *"Step 1: Register for the Challenge (Closed)"* |
| IJCAI 2025 Deepfake Detection & Localization (id 6891) | IJCAI 2025 | 2025-05-21 | **Closed** |
| General AIGC Audio–Video Detection (id 15769) | — | 2026-06-12 | **Closed** |
| Provenance Tracing in Sequential Deepfake Facial Edits (id 15351) | SIGIR 2026 | 2026-05-22 | **Closed** |
| NeurIPS 2025 Fairness in AI Face Detection (id 7166) | NeurIPS 2025 | 2025-10-31 | **Closed** |
| DFGC / DFGC-VRA | IJCB 2021, 2022, 2023 | 2023 | **Defunct.** `dfgc2021.iapr-tc4.org` loads but shows only 2021 content. No 2024/2025/2026 edition found in three targeted searches |
| Trusted Media Challenge | AI Singapore | 2021-12-15 | **Defunct.** Site loads and states *"CHALLENGE PERIOD 15 July – 15 December 2021"*. No successor |
| DFDC (Kaggle) | 2019-2020 | 2020 | **Closed.** No official "DFDC2" exists |
| Inclusion・Global Multimedia Deepfake Detection (`kaggle.com/competitions/multi-ffdi`) | Inclusion 2024 | 2024 | **Closed** — page returns 200 but is a JS shell; concluded per arXiv 2412.20833. **Partially verified** |

**The only genuinely live deepfake competition anywhere today is RTC-SDD** (Codabench id
17381) — Progress Evaluation Phase **2026-09-01 → 2026-11-09, `status: "Current"`**
**[FETCHED, I verified this directly]**, Final phase to 2026-11-16. **It is speech/audio,
not image.** Out of scope for you unless you pivot.

**COULD NOT VERIFY:** EvalAI's current listings (their search endpoint 404s); exact
open/closed dates for `kaggle.com/competitions/detect-ai-vs-human-generated-images`
(Kaggle blocks unauthenticated scraping and its REST API returns 401); the VASDL 2026
CVPR-workshop challenge (its page `dsri.org/challenges/vasdl-2026/` 404s); any DFGC
continuation past 2023.

---

## 7. Other benchmarks, briefly

| Benchmark | Live? | Size | Eval server? | Solo-GPU? |
|---|---|---|---|---|
| **OpenSDI** — *"OpenSDI: Spotting Diffusion-Generated Images in the Open World"*, CVPR 2025, arXiv 2503.19653 | **Yes.** HF `nebula/OpenSDI_train` (201K rows) + `nebula/OpenSDI_test`, CC-BY-SA-4.0. GitHub `iamwangyabin/OpenSDI` pushed 2026-06-18 | 201K rows | **Static author-run leaderboard** at `iamwangyabin.github.io/OpenSDI/` — not an eval server. Top: MaskCLIP, 0.8198 avg image-level acc | Likely yes for a subset. **Note: detection *plus* pixel-level localization** — a segmentation task, may not match your classifier |
| **DFBench** — ACM MM 2025, arXiv 2506.03007 | **Yes.** HF `IntMeGroup/DFBench`, Apache-2.0, lastModified 2025-06-11 | 540K images, **~121 GB in per-generator zips (1–14 GB each)** | No | **Yes** — the per-generator split makes this the most tractable large benchmark. Headline numbers COULD NOT VERIFY (results are in figures, not text) |
| **AIGCDetectBenchmark** — GitHub `Ekko-zn/AIGCDetectBenchmark`, tied to PatchCraft (arXiv 2311.12397) | **Yes.** pushed 2026-03-09, 436 stars | test set + checkpoints via Baidu/ModelScope | No — it is a *protocol + code* repo | Yes. Standard companion to GenImage; the "trained on ProGAN" table is widely cited (PatchCraft 89.31, AIDE 92.77 mean) |
| **SIDBench** — MAD '24 workshop, arXiv 2404.18552, Schinas & Papadopoulos | **Yes.** GitHub `mever-team/sidbench`, Apache-2.0, pushed 2026-07-30 | Code only — you supply `0_real`/`1_fake` dirs | No | **Yes, and useful** — 12 detectors integrated, weights on HF. Good for running baselines against your own data |
| **Community Forensics** — *"Community Forensics: Using Thousands of Generators to Train Fake Image Detectors"*, Park & Owens, **CVPR 2025**, arXiv 2411.04125 | **Yes.** HF `OwensLab/CommunityForensics`, CC-BY-4.0, lastModified 2025-10-24 | 2.76M images / 4,803 generators, **~1.08 TB** | No | Only via streaming or the **51.8K-image "PublicEval" subset** |
| **ELSA D3** (EU ELSA) | **Yes.** HF `elsaEU/ELSA_D3`, lastModified 2025-03-27, ~2.31M train rows | **~2.63 TB** | **No leaderboard found** | Streaming only. It is a *generation* dataset (SD1.4/2.1/SDXL/DeepFloyd IF), not a labelled real/fake benchmark |
| **FakeBench** — *"FakeBench: Probing Explainable Fake Image Detection via Large Multimodal Models"*, arXiv 2404.13306 | Partial. GitHub `Yixuanli423/FakeBench` updated 2026-08-29 | 6,000-image pool | No | **Labels are deliberately withheld** ("to avoid corpus leakage"). Not a clean labelled benchmark. LMM-oriented — probably not your task. TIFS venue is [SNIPPET] only |
| **DeepfakeBench** — NeurIPS 2023 D&B, GitHub `SCLBD/DeepfakeBench` | **Yes.** pushed 2025-08-20, **1,102 stars**, not archived | Code framework | No leaderboard | **Yes — strongly worth your time.** 36 detectors (28 image, 8 video), 9 datasets, standardised protocols. The most credible way to produce numbers that reviewers accept as comparable, without an eval server |
| **HF "Deepfake Detection Arena" Space** (`bitmind/dfd-arena-leaderboard`) | Shows "Running", but README last commit ~2 years ago, no 2025/26 activity | — | Nominally has a submission queue | **Treat as dormant.** Not a reliable external reference |

---

## What I would actually do

1. **Submit to FF++ this week.** It is 565 MB and one inference pass. The infrastructure is
   verifiably up and the sibling benchmark proves the scorer generally works. Accept that
   the entry may never appear, and do not build the thesis around it. Dry-run the JSON
   locally first — the 2-week block means a malformed submission costs you a fortnight.
2. **Apply for Deepfake-Eval-2024 access today**, because approval is manual and slow. It
   is the strongest *current* external reference: recent, in-the-wild, small, with
   published baselines your zero-shot number slots directly against. Its whole finding —
   detectors lose ~45% AUC in the wild — is the same story your own generalisation result
   tells, which makes it the natural anchor for that chapter.
3. **Email for Chameleon** in parallel. Costs one email. If it arrives, the near-zero
   fake-class accuracy across all ten published detectors is the most quotable result in
   this space.
4. **Run DeepfakeBench** for your in-domain comparisons. Standardised protocol beats an
   abandoned leaderboard for reviewer credibility.
5. **Calendar: early January 2027**, NTIRE / CVPR AI-generated image detection. That is the
   real live competition — you missed it by three days.

## Corrections to existing docs

- `docs/EXTERNAL_BENCHMARK.md` step 1 says leaderboard entries *"appear to stop around
  2024"*. They stop around **2021-2022** — the board has held at 109 entries since
  2022-10-06.
- Its rate-limit note (*"Submissions are usually rate-limited per method"*) is correct and
  can now be stated precisely: two weeks, per the benchmark's own wording.
- The 0.973 top score and the `.zip`/`.json`/`"fake"`/`"real"` submission format are both
  confirmed correct.
