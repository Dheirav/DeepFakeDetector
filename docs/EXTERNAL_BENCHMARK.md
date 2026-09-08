# External benchmark: submitting to FaceForensics++

## Why

The headline number in this repo — **89.5% validation accuracy** (run 21,
`convnext-small__light__0.4__cosine__focal__srm`) — is measured on a split this project
built itself. The dataset construction is careful, with perceptual-hash dedup, quality
filtering and leakage-resistant cluster splitting, but it is still self-graded. Nobody can
compare 89.5% here against 89.5% anywhere else.

FaceForensics++ has a **hidden test set and a public leaderboard**, so a score there is
directly comparable to published papers. Top of the board is currently around 0.973.

Benchmark: https://kaldir.vc.in.tum.de/faceforensics_benchmark/
Paper: https://arxiv.org/abs/1901.08971

## Submission format

A single `.zip` or `.7z` containing one `.json` file: a dictionary mapping every benchmark
image filename to one of exactly two labels, `"fake"` or `"real"`.

## The mapping problem — read this first

This project is a **3-class** classifier: real / AI-generated / AI-edited. The benchmark is
**binary**. The obvious collapse is `AI-generated ∪ AI-edited → fake`, and that is what to
submit, but it is worth checking whether the collapsed binary decision is better made by
summing the two fake-class probabilities or by taking the argmax over three classes. They
are not the same decision rule and one of them will be measurably better. Test that on the
local held-out split before spending a submission on it.

## Expect a domain gap, and report it honestly

FF++ is **face manipulation in video frames** (Deepfakes, Face2Face, FaceSwap,
NeuralTextures). The dataset here spans 20 sources of general AI-generated and AI-edited
imagery, which is a broader and different distribution. The score will very likely drop
against the 89.5% in-domain figure.

That drop is a result, not a failure, and it is more interesting than the in-domain number.
It measures whether the forensic signals this project bet on — SRM residuals, FFT frequency
analysis, CBAM attention — transfer to a manipulation family they were never trained on.
Given that the ablation already showed those components buy under 0.2% in-domain, the
cross-domain result is where they either justify themselves or do not.

Report zero-shot transfer first, before any fine-tuning on FF++ data. That is the honest
measurement and it is the one worth writing down.

## Steps

1. ~~Confirm the benchmark is still accepting submissions.~~ **Done 2026-09-08 — see
   `docs/BENCHMARK_LIVENESS_2026-09-08.md`.** Summary: the server, registration form,
   565 MB image download and submission docs are all live and were fetched successfully.
   But the leaderboard has been frozen at exactly **109 entries since 2022-10-06** — the
   "stop around 2024" guess above was wrong, it is 2021-2022. The ScanNet benchmark on the
   same server and codebase gained ~1,270 cells in the last nine months, so the TUM
   infrastructure is alive and FF++ is community-dead rather than server-dead. Whether the
   FF++ scorer still returns a result is unproven without spending a submission.
2. Download the benchmark image set (the public images, not the labels — labels are hidden).
3. Pick the checkpoint. Run 21 is the best on validation; check the model card in
   `model_cards/` for what it was trained with.
4. Run inference with the same preprocessing used at training time. Mismatched preprocessing
   is the most common way to lose several points for no reason.
5. Collapse to binary using whichever rule tested better locally.
6. Emit the JSON, zip it, submit.
7. Record the returned score in `results/` alongside the local numbers, with a note on the
   domain gap.

## Watch out for

- Submissions are rate-limited per method — **verified 2026-09-08**, the benchmark's own
  wording is: *"we block updates to the test set results of a method for two weeks after a
  test set submission."* Two weeks. Do not burn one on an untested preprocessing pipeline.
  Dry-run the whole path on local data first and confirm the JSON parses and covers every
  filename. The official example submission is 1,000 entries of `{"0000.png": "fake", ...}`.
- Registering with more than one e-mail address is grounds for a ban: *"We will ban users
  or domains if required."*
- Do not quote an FF++ number and the local 89.5% as if they measure the same thing.
