"""Cascade inference helper.

Provides a small `CascadeClassifier` that accepts two pre-loaded models
and runs the two-stage logic described in the project requirements.

Stage-1 classes are expected to follow: 0=Real, 1=AI-Generated, 2=AI-Edited
Stage-2 is a binary classifier: 0=Real, 1=AI-Edited
"""
from typing import Optional, Dict, Any
import numpy as np
import torch
import torch.nn.functional as F


class CascadeClassifier:
    def __init__(self, stage1_model: torch.nn.Module, stage2_model: torch.nn.Module, device: str = "cpu", cascade_threshold: Optional[float] = None):
        """Create a cascade runner from two already-loaded models.

        Args:
            stage1_model: 3-class model (0=Real,1=AI-Generated,2=AI-Edited)
            stage2_model: 2-class model (0=Real,1=AI-Edited)
            device: device string ("cpu" or "cuda")
            cascade_threshold: When provided, run Stage-2 only if
                |P_stage1(Real) - P_stage1(AI-Edited)| <= cascade_threshold.
                If None, Stage-2 is always run for samples routed to it.
        """
        self.stage1 = stage1_model.to(device)
        self.stage2 = stage2_model.to(device)
        self.device = device
        self.threshold = cascade_threshold

        # ensure eval mode
        self.stage1.eval()
        self.stage2.eval()

    def run_on_dataloader(self, dataloader) -> Dict[str, Any]:
        """Run cascade on a DataLoader producing (image, label) pairs.

        Returns a dict with:
            y_true: np.array
            y_pred: np.array (final labels in 0/1/2 space)
            stage1_preds: np.array
            stage1_probs: np.array (Nx3)
            stage2_preds: np.array (NaN for examples where not run)
            stage2_probs: np.array (Nx2, NaN rows where not run)
            stats: dict with forwarding counts
        """
        all_labels = []
        all_stage1_preds = []
        all_stage1_probs = []
        # placeholders for stage2 outputs
        all_stage2_preds = []
        all_stage2_probs = []

        # First pass: run stage1 on all samples and collect tensors and labels
        images_buffer = []
        labels_buffer = []
        idx_map = []

        with torch.no_grad():
            for batch_images, batch_labels in dataloader:
                batch_images = batch_images.to(self.device)
                outputs = self.stage1(batch_images)
                probs = F.softmax(outputs, dim=1).cpu().numpy()
                preds = probs.argmax(axis=1)

                all_stage1_probs.extend(probs.tolist())
                all_stage1_preds.extend(preds.tolist())
                all_labels.extend(batch_labels.numpy().tolist())

                # store image tensors for possible stage2 rerun
                images_buffer.append(batch_images.cpu())
                labels_buffer.extend(batch_labels.numpy().tolist())

        # concat buffers
        all_stage1_probs = np.array(all_stage1_probs)
        all_stage1_preds = np.array(all_stage1_preds)
        y_true = np.array(all_labels)

        N = len(y_true)
        all_stage2_preds = np.full((N,), np.nan)
        all_stage2_probs = np.full((N, 2), np.nan)

        # Determine indices requiring Stage-2: stage1 predicted Real(0) or AI-Edited(2)
        candidates = np.where((all_stage1_preds == 0) | (all_stage1_preds == 2))[0]

        # apply threshold gating if provided
        if self.threshold is not None:
            # delta between Real and AI-Edited probabilities
            real_probs = all_stage1_probs[:, 0]
            edited_probs = all_stage1_probs[:, 2]
            deltas = np.abs(real_probs - edited_probs)
            gated = deltas <= self.threshold
            # keep only candidates where gated is True
            candidates = [int(i) for i in candidates if gated[i]]

        # If there are no candidates, skip Stage-2 pass
        if len(candidates) > 0:
            # create a flat list of all images again in the same order
            all_images = torch.cat([b for b in images_buffer], dim=0)

            # Run stage2 in batches for selected indices
            batch_size = dataloader.batch_size if hasattr(dataloader, 'batch_size') else 64
            # gather candidate images
            cand_imgs = all_images[candidates].to(self.device)
            with torch.no_grad():
                for offset in range(0, len(candidates), batch_size):
                    sub = cand_imgs[offset: offset + batch_size]
                    out2 = self.stage2(sub)
                    probs2 = F.softmax(out2, dim=1).cpu().numpy()
                    preds2 = probs2.argmax(axis=1)
                    for i, idx in enumerate(candidates[offset: offset + batch_size]):
                        all_stage2_preds[idx] = int(preds2[i])
                        all_stage2_probs[idx, :] = probs2[i]

        # Combine final predictions: start from stage1 and override where stage2 ran
        final_preds = all_stage1_preds.copy()
        # for indices where stage2 produced a non-nan prediction, map stage2 {0->0,1->2}
        ran_stage2 = ~np.isnan(all_stage2_preds)
        final_preds[ran_stage2] = np.array([0 if p == 0 else 2 for p in all_stage2_preds[ran_stage2].astype(int)])

        stats = {
            'n_samples': int(N),
            'n_candidates': int(len(candidates)),
            'n_stage2_run': int(ran_stage2.sum()),
            'cascade_threshold': float(self.threshold) if self.threshold is not None else None,
        }

        return {
            'y_true': y_true,
            'y_pred': final_preds.astype(int),
            'stage1_preds': all_stage1_preds.astype(int),
            'stage1_probs': all_stage1_probs,
            'stage2_preds': all_stage2_preds,
            'stage2_probs': all_stage2_probs,
            'stats': stats,
        }
