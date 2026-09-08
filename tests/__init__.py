"""Test package.

Doubles as the path setup: this runs before any test module is imported under
`python -m unittest discover`, where conftest.py (a pytest convention) is not
consulted. Keeping both means either runner works.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (ROOT, os.path.join(ROOT, "scripts"), os.path.join(ROOT, "dataset_builder")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
