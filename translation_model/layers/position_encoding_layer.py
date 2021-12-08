import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer, Dropout


class PositionEncodingLayer(Layer):

    def __init__(self, dim, encoding_length, dropout_rate=0.1):
        super(PositionEncodingLayer, self).__init__()
        self.dim = dim
        self.length = encoding_length
        self.pe_layer = self._build_layer()
        self.dropout_layer = Dropout(rate=dropout_rate)

    def _build_layer(self):
        encoding = np.zeros((self.length, self.dim))
        num = np.expand_dims(np.arange(self.length), axis=1)
        den = np.power(10000, 2 * np.arange(0, 2 * (self.dim // 2)) / np.float32(self.dim))
        angle = num / tf.cast(den, tf.int32)

        encoding[:, 0::2] = np.sin(angle[:, 0::2])
        encoding[:, 1::2] = np.cos(angle[:, 1::2])

        return tf.cast(np.expand_dims(encoding, axis=0), dtype=tf.float32)

    def get_config(self):
        config = super(PositionEncodingLayer, self).get_config()
        config.update({
            'dim': self.dim,
            'length': self.length,
            'dropout_layer': self.dropout_layer
        })

        return config

    def call(self, inputs, in_training):
        input_len = inputs.shape[1]
        #tf.print(f'Input length {input_len} and {inputs.shape}')
        inputs = inputs + self.pe_layer[:, :input_len, :]
        # tf.print(f'Input is now {inputs.shape}')

        return self.dropout_layer(inputs, training=in_training)
