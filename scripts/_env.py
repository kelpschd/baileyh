"""
Environment setup that MUST run before cellpose is imported anywhere.

Import this module FIRST (before `from cellpose import ...`) in any module that
touches cellpose — currently only segmentation.py. Importing it runs the
os.environ.setdefault below as a side effect.

setdefault means a value set externally (e.g. exported in the sbatch script or
by biowulf_runner.py) still wins; this is only the fallback.
"""
import os

os.environ.setdefault("CELLPOSE_LOCAL_MODELS_PATH", "/data/kelpschdj/cellpose/models")