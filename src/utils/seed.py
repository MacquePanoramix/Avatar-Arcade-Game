"""Reproducibility helper for Python, NumPy, and TensorFlow."""

import os
import random

import numpy as np
import tensorflow as tf


def set_global_seed(seed: int = 42) -> None:
    """Set random seeds across common ML libraries."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
