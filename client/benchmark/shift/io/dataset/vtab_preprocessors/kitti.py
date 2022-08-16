"""
Adapted from task adaptation project.
"""

import numpy as np
import tensorflow as tf

def _closest_vehicle_distance_pp(x):
  """Predict the distance to the closest vehicle."""
  # Label distribution:

  # Location feature contains (x, y, z) in meters w.r.t. the camera.
  vehicles = tf.where(x["objects"]["type"] < 3)  # Car, Van, Truck.
  vehicle_z = tf.gather(params=x["objects"]["location"][:, 2], indices=vehicles)
  vehicle_z = tf.concat([vehicle_z, tf.constant([[1000.0]])], axis=0)
  dist = tf.reduce_min(vehicle_z)
  # Results in a uniform distribution over three distances, plus one class for
  # "no vehicle".
  thrs = np.array([-100.0, 8.0, 20.0, 999.0])
  label = tf.reduce_max(tf.where((thrs - dist) < 0))
  return x["image"], label