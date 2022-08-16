import tensorflow as tf

def dsprites_location_fn(x):
    image = tf.tile(x["image"], [1, 1, 3]) * 255
    label = tf.cast(
        tf.math.floordiv(tf.cast(x['label_x_position'], tf.float32), 32.0 / 16),
        tf.int64,
    )
    return image, label

def dsprites_orientation_fn(x):
    image = tf.tile(x["image"], [1, 1, 3]) * 255
    label = tf.cast(
        tf.math.floordiv(tf.cast(x['label_orientation'], tf.float32), 40.0 / 16),
        tf.int64,
    )
    return image, label