from ..utils import deepcopy_n
from .encoder_layer import EncoderLayer
from .position_encoding_layer import PositionEncodingLayer

import tensorflow as tf
from tensorflow.keras.layers import Layer, Embedding


class Encoder(Layer):

    def __init__(self, num_enc_layers, dim, heads, inner_dim, seq_len, enc_vocab_size,
                 pe_size, dropout_rate=0.1):
        super(Encoder, self).__init__()
        self.num_enc_layers = num_enc_layers
        self.dim = dim
        self.heads = heads
        self.inner_dim = inner_dim
        self.seq_len = seq_len

        self.embedding = Embedding(input_dim=enc_vocab_size, output_dim=dim)
        self.position_encoding = PositionEncodingLayer(dim, encoding_length=pe_size,
                                                       dropout_rate=dropout_rate)

        self.encoder_layers = deepcopy_n(EncoderLayer(dim,
                                                      heads=heads,
                                                      inner_dim=inner_dim,
                                                      seq_len=seq_len,
                                                      dropout_rate=dropout_rate,
                                                      epsilon=1e-6), times=num_enc_layers)

    def get_config(self):
        config = super(Encoder, self).get_config()
        config.update({
            'num_enc_layers': self.num_enc_layers,
            'dim': self.dim,
            'inner_dim': self.inner_dim,
            'heads': self.heads,
            'seq_len': self.seq_len
        })
        return config

    def call(self, inputs, in_training, mask):
        scale = tf.sqrt(tf.cast(self.dim, dtype=tf.float32))
        # BATCHES * REPLICAS, inputs_seq_len -> inputs

        inputs = self.embedding(inputs) * scale
        # BATCHES * REPLICAS, inputs_seq_len, dim -> inputs
        inputs = self.position_encoding(inputs, in_training=in_training)
        # BATCHES * REPLICAS, inputs_seq_len, dim -> inputs

        for i in range(self.num_enc_layers):
            inputs = self.encoder_layers[i](inputs, in_training=in_training, mask=mask)

        return inputs
