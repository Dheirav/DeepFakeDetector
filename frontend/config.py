import os
from typing import Optional

# Configurable model checkpoint path (can be overridden by env var)
MODEL_CHECKPOINT: Optional[str] = os.environ.get(
    "MODEL_CHECKPOINT",
    # Run 17, not run 19. Run 19 scores higher on validation and collapses to 0.000
    # on AI-generated images downscaled to 256px and re-saved at JPEG q60; run 17
    # holds 0.955 on the same images. Validation accuracy is inversely correlated
    # (r = -0.956) with robustness here -- see LIMITATIONS.md.
    "models/17__convnext-small__strong__0.4__cosine__focal__srm-gem/best_model.pth"
)

# Allow user to force CPU by setting env var USE_CPU=1
USE_CPU_ENV = os.environ.get("USE_CPU", "0")
