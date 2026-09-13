import os
from typing import Optional

# Configurable model checkpoint path (can be overridden by env var)
MODEL_CHECKPOINT: Optional[str] = os.environ.get(
    "MODEL_CHECKPOINT",
    # The frozen-encoder CLIP mask head trained on the verified-clean OpenSDI
    # slice: 0.8040 balanced in-domain, mean held-out-generator recall 0.723.
    # Not results/mask_head_clip448_ft4 (0.9015 in-domain), because fine-tuning
    # the encoder cost 7 points of transfer and that is the number that matters
    # for an image of unknown origin. Not models/17 either: that ConvNeXt reads
    # file provenance, not content; see LIMITATIONS.md sections 1 to 8.
    "results/mask_head_clip448_balanced/best_model.pth"
)

# The confounded original, kept loadable so the two can be compared on the
# same upload. Run 17 rather than 19: 19 collapses to 0.000 on re-saved images.
LEGACY_CHECKPOINT = "models/17__convnext-small__strong__0.4__cosine__focal__srm-gem/best_model.pth"

# Allow user to force CPU by setting env var USE_CPU=1
USE_CPU_ENV = os.environ.get("USE_CPU", "0")
