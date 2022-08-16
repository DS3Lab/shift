"""
Adapted from task adaptation project.
"""

import numpy as np
import tensorflow as tf

def _closest_object_preprocess_fn(x):
  dist = tf.reduce_min(x["objects"]["pixel_coords"][:, 2])
  # These thresholds are uniformly spaced and result in more or less balanced
  # distribution of classes, see the resulting histogram:

  thrs = np.array([0.0, 8.0, 8.5, 9.0, 9.5, 10.0, 100.0])
  label = tf.reduce_max(tf.where((thrs - dist) < 0))
  return x["image"], label

def _count_preprocess_fn(x):
  return x["image"], tf.size(x["objects"]["size"]) - 3
