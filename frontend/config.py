import os
from typing import Optional

# Configurable model checkpoint path (can be overridden by env var)
MODEL_CHECKPOINT: Optional[str] = os.environ.get("MODEL_CHECKPOINT", "models/best_model.pt")

# Allow user to force CPU by setting env var USE_CPU=1
USE_CPU_ENV = os.environ.get("USE_CPU", "0")
