"""Preprocessing package: augmentation pipelines, SRM residuals, FFT magnitude.

This file exists to make the directory a *regular* package. Without it Python
treats it as a namespace portion, which loses to any real module of the same
name found on the path -- so `scripts/preprocessing/preprocessing.py` shadowed
the package whenever a script inside this directory was run, and
`from preprocessing import preprocessing` raised ImportError.
"""
