from copy import deepcopy
from tensorflow.keras.layers import Layer, Dense, Dropout, MultiHeadAttention, LayerNormalization
from tensorflow.keras import Sequential


class EncoderLayer(Layer):

    def __init__(self, dim, heads, inner_dim, seq_len, dropout_rate=0.1, epsilon=1e-6):
        super(EncoderLayer, self).__init__()
        self.dim = dim
        self.heads = heads
        self.dropout_rate = dropout_rate
        self.epsilon = epsilon
        self.inner_dim = inner_dim
        self.seq_len = seq_len

        self.multi_head_attention = MultiHeadAttention(num_heads=heads,
                                                       key_dim=dim,
                                                       dropout=dropout_rate)

        layers = [
            Dense(inner_dim, activation='relu'),
            Dense(dim),
            Dropout(rate=dropout_rate)
        ]

        self.feed_forward_network = Sequential(layers)

        self.layer_norm = LayerNormalization(epsilon=epsilon)
        self.layer_norm2 = deepcopy(self.layer_norm)

    def get_config(self):
        config = super(EncoderLayer, self).get_config()
        config.update({
            'epsilon': self.epsilon,
            'dim': self.dim,
            'inner_dim': self.inner_dim,
            'heads': self.heads,
            'seq_len': self.seq_len,
            'dropout_rate': self.dropout_rate
        })

        return config

    def call(self, inputs, in_training, mask):
        attention = self.multi_head_attention(inputs, value=inputs, key=inputs,
                                              attention_mask=mask,
                                              training=in_training)

        attention = self.layer_norm(inputs + attention)

        feed_fwd_out = self.feed_forward_network(attention, training=in_training)
        feed_fwd_out = self.layer_norm2(attention + feed_fwd_out)

        return feed_fwd_out

