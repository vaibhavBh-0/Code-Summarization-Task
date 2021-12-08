import tensorflow as tf
from tensorflow.keras.optimizers.schedules import LearningRateSchedule


class Scheduler(LearningRateSchedule):
    """
    Learning Rate Scheduler implemented to match the one used in the original Transformer paper.
    """

    def __init__(self, dim, warmup_steps=4000.0):
        super(Scheduler, self).__init__()
        dtype = tf.float32
        self.dim = tf.cast(dim, dtype=dtype)
        self.warmup_steps = warmup_steps

        self.exp = tf.cast(-0.5, dtype=dtype)
        self.exp_2 = tf.cast(-1.5, dtype=dtype)

    def __call__(self, step):
        scale = tf.pow(self.dim, self.exp)
        left = tf.pow(step, self.exp)
        right = step * tf.pow(self.warmup_steps, self.exp_2)

        return scale * tf.minimum(left, right)
