from copy import deepcopy
from ..utils.utils import deepcopy_n

from tensorflow.keras.layers import Layer, Dense, Dropout, MultiHeadAttention, LayerNormalization
from tensorflow.keras import Sequential
import tensorflow as tf


class DecoderLayer(Layer):

    def __init__(self, dim, heads, inner_dim, seq_len, dropout_rate=0.1, epsilon=1e-6):
        super(DecoderLayer, self).__init__()
        self.dim = dim
        self.heads = heads
        self.dropout_rate = dropout_rate
        self.epsilon = epsilon
        self.inner_dim = inner_dim
        self.enc_seq_len = seq_len
        self.dec_seq_len = seq_len - 1

        self.multi_head_attention = MultiHeadAttention(num_heads=heads,
                                                       key_dim=dim,
                                                       dropout=dropout_rate)

        self.multi_head_attention2 = deepcopy(self.multi_head_attention)

        layers = [
            Dense(inner_dim, activation='relu'),
            Dense(dim),
            Dropout(rate=dropout_rate)
        ]

        self.feed_forward_network = Sequential(layers)

        layers = deepcopy_n(LayerNormalization(epsilon=epsilon), times=3)
        self.layer_norm, self.layer_norm2, self.layer_norm3 = layers

    def get_config(self):
        config = super(DecoderLayer, self).get_config()
        config.update({
            'epsilon': self.epsilon,
            'dim': self.dim,
            'inner_dim': self.inner_dim,
            'heads': self.heads,
            'enc_seq_len': self.enc_seq_len,
            'dec_seq_len': self.dec_seq_len,
            'dropout_rate': self.dropout_rate
        })

        return config

    def call(self, inputs, encoder_out, in_training, look_ahead_mask, padding_mask):

        encoder_out = tf.ensure_shape(encoder_out, [None, self.enc_seq_len, self.dim])

        attention = self.multi_head_attention(inputs, value=inputs, key=inputs,
                                              attention_mask=look_ahead_mask,
                                              training=in_training)

        attention = tf.ensure_shape(attention, [None, self.dec_seq_len, self.dim])

        attention = self.layer_norm(inputs + attention)

        attention = tf.ensure_shape(attention, [None, self.dec_seq_len, self.dim])

        attention_on_enc = self.multi_head_attention2(attention, value=encoder_out,
                                                      key=encoder_out,
                                                      attention_mask=padding_mask,
                                                      training=in_training)

        attention_on_enc = self.layer_norm2(attention + attention_on_enc)

        attention_on_enc = tf.ensure_shape(attention_on_enc, [None, self.dec_seq_len, self.dim])

        feed_fwd_out = self.feed_forward_network(attention_on_enc, training=in_training)
        feed_fwd_out = self.layer_norm3(attention_on_enc + feed_fwd_out)

        return feed_fwd_out

