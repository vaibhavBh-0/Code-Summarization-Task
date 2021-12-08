from ..utils.utils import deepcopy_n
from .decoder_layer import DecoderLayer
from .position_encoding_layer import PositionEncodingLayer

import tensorflow as tf
from tensorflow.keras.layers import Layer, Embedding


class Decoder(Layer):

    def __init__(self, num_dec_layers, dim, heads, inner_dim, seq_len, dec_vocab_size, pe_size, dropout_rate=0.1):
        super(Decoder, self).__init__()

        self.num_dec_layers = num_dec_layers
        self.dim = dim
        self.heads = heads
        self.inner_dim = inner_dim
        self.seq_len = seq_len

        self.embedding = Embedding(input_dim=dec_vocab_size, output_dim=dim)
        self.position_encoding = PositionEncodingLayer(dim, encoding_length=pe_size,
                                                       dropout_rate=dropout_rate)

        self.decoder_layers = deepcopy_n(DecoderLayer(dim,
                                                      heads=heads,
                                                      inner_dim=inner_dim,
                                                      seq_len=seq_len,
                                                      dropout_rate=dropout_rate,
                                                      epsilon=1e-6), times=num_dec_layers)

    def get_config(self):
        config = super(Decoder, self).get_config()
        config.update({
            'num_dec_layers': self.num_dec_layers,
            'dim': self.dim,
            'inner_dim': self.inner_dim,
            'heads': self.heads,
            'seq_len': self.seq_len
        })
        return config

    def call(self, inputs, encoder_out, in_training, look_ahead_mask, padding_mask):
        scale = tf.sqrt(tf.cast(self.dim, dtype=tf.float32))
        # BATCHES * REPLICAS, target_seq_len, dim -> inputs
        # BATCHES * REPLICAS, input_seq_len, dim -> encoder_out

        inputs = self.embedding(inputs) * scale
        inputs = self.position_encoding(inputs, in_training=in_training)

        for i in range(self.num_dec_layers):
            inputs = self.decoder_layers[i](inputs, encoder_out=encoder_out, in_training=in_training,
                                            look_ahead_mask=look_ahead_mask, padding_mask=padding_mask)

        return inputs
