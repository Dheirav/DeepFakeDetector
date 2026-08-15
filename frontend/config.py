import os
from typing import Optional

# Configurable model checkpoint path (can be overridden by env var)
MODEL_CHECKPOINT: Optional[str] = os.environ.get(
    "MODEL_CHECKPOINT",
    "models/19__convnext-small__light__0.4__cosine__focal__none/best_model.pth"
)

# Allow user to force CPU by setting env var USE_CPU=1
USE_CPU_ENV = os.environ.get("USE_CPU", "0")
