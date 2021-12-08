from tensorflow.keras.optimizers import Adam
from translation_model import TransformerModel, Scheduler
import numpy as np

num_layers = 2
dim = 256
inner_dim = 512
heads = 4
dropout_rate = 0.1
MAX_INPUT_LEN = 1024

EPOCHS = 20
BATCHES = 8 * 2

VOCAB_SIZE = 33000
START_TOKEN = 2
END_TOKEN = 3


def get_model():
    scheduler = Scheduler(dim)
    optimizer = Adam(scheduler, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

    model = TransformerModel(num_layers, dim=dim, heads=heads, inner_dim=inner_dim, seq_len=MAX_INPUT_LEN,
                             enc_vocab_size=VOCAB_SIZE, dec_vocab_size=VOCAB_SIZE, pe_enc_size=MAX_INPUT_LEN,
                             pe_dec_size=MAX_INPUT_LEN, dropout_rate=dropout_rate)

    model.build(input_shape=[(None, MAX_INPUT_LEN), (None, MAX_INPUT_LEN - 1)])
    model.summary()

    return model, optimizer


if __name__ == '__main__':
    model1, optimizer1 = get_model()
    model2, optimizer2 = get_model()

    for model1_wt, model2_wt in zip(model1.weights, model2.weights):
        assert ~ np.allclose(model1_wt.numpy(), model2_wt.numpy())

    print('Model weights are dissimilar')

    for op1_wt, op2_wt in zip(optimizer1.weights, optimizer2.weights):
        assert ~ np.allclose(op1_wt.numpy(), op2_wt.numpy())

    print('Optimizer weights are dissimilar')

    model1.save_weights('model1_weights')

    model2.load_weights('model1_weights')

    for model1_wt, model2_wt in zip(model1.weights, model2.weights):
        assert np.allclose(model1_wt.numpy(), model2_wt.numpy())

    print('Model weights are now same')




