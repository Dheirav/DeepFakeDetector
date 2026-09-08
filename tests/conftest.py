"""Make the project's packages importable without installing anything."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (ROOT, os.path.join(ROOT, "scripts"), os.path.join(ROOT, "dataset_builder")):
    if p not in sys.path:
        sys.path.insert(0, p)
