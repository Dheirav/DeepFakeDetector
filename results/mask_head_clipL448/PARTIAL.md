# CLIP ViT-L/14 @448 — partial run, stopped deliberately

**Status: incomplete. One epoch of eight. Do not quote these as a trained result.**

## What it was testing

DINOv2 ViT-S held the best mask IoU (0.277) despite the worst classification,
and the only structural difference from CLIP ViT-B/16 was the patch grid: 32x32
against 28x28. CLIP ViT-L/14 has a 14px patch, so at 448px it gives the same
32x32 grid, isolating grid resolution from feature quality.

## What one epoch showed

    epoch 1/8   acc 0.8286   ai_edited F1 0.4671   IoU 0.2546

Against CLIP ViT-B/16 @448, whose best across all ten of its epochs was
**IoU 0.237**. A single epoch of the 32x32 model beat ten epochs of the 28x28
one on the metric the experiment was about, so the direction is established:
**grid resolution helps localisation independently of feature quality.**

The magnitude is not established. One epoch is not a converged model, and
ViT-L/14 also brings roughly three times the parameters of ViT-B/16, so grid
size and capacity are confounded here and cannot be separated from this run.

## Why it was stopped

Measured 114 to 143 minutes per epoch, putting the remaining seven epochs at 13
to 17 hours. The GPU ran at 100% utilisation and climbed from 82C to 84C, which
on a laptop card means progressive thermal throttling.

A throughput benchmark before the run predicted 31 min/epoch. That benchmark
timed 5 batches after 2 warmup, roughly 3.5 seconds of GPU work, which is far too
short to reach the sustained thermal state of a multi-hour run. **It measured
peak throughput and was reported as sustained throughput.** Any future estimate
on this hardware should benchmark for minutes, not seconds.

Sixteen hours to refine a decimal place on a conclusion already established was
not a good trade, so the run was stopped by choice rather than by failure.

## What survives

`best_model.pth` holds the epoch-1 decoder and classifier weights. The encoder is
not saved, since it is frozen and reloadable from torch.hub.

No `y_true.npy`, `y_pred.npy`, `probs.npy` or `training_summary.json`, because
those are written after the final epoch. The numbers above come from the training
log and were computed on the same held-out split as every other run here.

## If resumed

Four epochs would take roughly 5.7 hours and would likely capture the peak, since
every run in this project converges by epoch 3 and best epochs land between 7 and
12. Reducing the per-epoch evaluation, currently a full pass over 6,353 test
images every epoch, would cut that further.
