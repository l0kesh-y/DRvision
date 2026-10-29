"""
view_metrics.py – View Model Accuracy, F1-Score, QWK & Per-Class Metrics in Terminal.

Usage:
    python view_metrics.py
"""

import sys
import os

# Ensure the root directory is on the python search path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from ml.densenet_model import print_terminal_metrics

if __name__ == "__main__":
    print_terminal_metrics()
