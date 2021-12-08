import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense

from ..layers import Encoder, Decoder

tnp = tf.experimental.numpy
tnp.experimental_enable_numpy_behavior()


class TransformerModel(Model):

    def __init__(self, num_layers, heads, dim, inner_dim, seq_len, enc_vocab_size, dec_vocab_size, pe_enc_size,
                 pe_dec_size, dropout_rate=0.1):
        super(TransformerModel, self).__init__()
        self.inner_dim = inner_dim
        self.dim = dim
        self.dec_vocab_size = dec_vocab_size
        self.seq_len = seq_len

        self.encoder = Encoder(num_enc_layers=num_layers, dim=dim, heads=heads, seq_len=seq_len, inner_dim=inner_dim,
                               enc_vocab_size=enc_vocab_size, pe_size=pe_enc_size, dropout_rate=dropout_rate)

        self.decoder = Decoder(num_dec_layers=num_layers, dim=dim, heads=heads, seq_len=seq_len, inner_dim=inner_dim,
                               dec_vocab_size=dec_vocab_size, pe_size=pe_dec_size, dropout_rate=dropout_rate)

        self.output_layer = Dense(dec_vocab_size)

    def get_config(self):
        base_config = {
            'inner_dim': self.inner_dim,
            'dim': self.dim,
            'dec_vocab_size': self.dec_vocab_size,
            'seq_len': self.seq_len
        }
        encoder_config = self.encoder.get_config()
        decoder_config = self.decoder.get_config()

        return dict(list(base_config.items()) + list(encoder_config.items()) + list(decoder_config.items()))

    @classmethod
    def _get_padding_mask(cls, inputs):
        mask = tf.cast(inputs == 0, tf.float32)
        mask = mask[:, tf.newaxis, tf.newaxis, :]

        return mask

    @classmethod
    def _get_look_ahead_masks(cls, dim):
        # Use tf experimental numpy to increase performance on GPU/TPUs.

        return tf.cast(tnp.triu(tnp.ones((dim, dim)), k=1), dtype=tf.float32)

    def _get_masks(self, inputs, target, target_length):
        enc_mask = self._get_padding_mask(inputs)
        # TODO: Check if deepcopy affects TPU perf.
        dec_mask = self._get_padding_mask(inputs)

        look_ahead_mask = self._get_look_ahead_masks(target_length)
        dec_target_mask = self._get_padding_mask(target)
        look_ahead_mask = tf.maximum(look_ahead_mask, dec_target_mask)

        return enc_mask, dec_mask, look_ahead_mask

    def call(self, inputs, training):
        input = inputs[0]
        target = inputs[1]

        input_seq_len = self.seq_len
        target_seq_len = self.seq_len - 1

        input = tf.ensure_shape(input, [None, input_seq_len])
        target = tf.ensure_shape(target, [None, target_seq_len])

        enc_mask, dec_mask, look_ahead_mask = self._get_masks(input, target, target_length=target_seq_len)

        encoder_out = self.encoder(input, in_training=training, mask=enc_mask)
        encoder_out = tf.ensure_shape(encoder_out, [None, input_seq_len, self.dim])

        decoder_out = self.decoder(target, encoder_out=encoder_out, in_training=training,
                                   look_ahead_mask=look_ahead_mask, padding_mask=dec_mask)
        decoder_out = tf.ensure_shape(decoder_out, [None, target_seq_len, self.dim])

        output = self.output_layer(decoder_out)
        output = tf.ensure_shape(output, [None, target_seq_len, self.dec_vocab_size])

        return output
