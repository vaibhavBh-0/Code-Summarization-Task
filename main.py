from time import time
import tensorflow as tf
import tensorflow_text as text
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Mean
from translation_model import Scheduler, TransformerModel
import pandas as pd
from prerequisites import ensure_dataset_is_downloaded, randomize_dataset, resolve_tpu_strategy, export_csv

file_name = 'python_dataset.pkl'
Dataset = tf.data.Dataset

# Name of the TPU Node on Google Cloud.
TPU_ADDRESS = 'cs421-tpu'
# Bucket where the processed data is hosted. Files will be written and read from here.
BUCKET = 'gs://code-search-python-dataset/'
checkpoint_path = f'{BUCKET}model'
model_history_path = f'{checkpoint_path}/history/'

num_layers = 2
dim = 256
inner_dim = 512
heads = 4
dropout_rate = 0.1
MAX_INPUT_LEN = 1024

EPOCHS = 20
# number of batches to be iterated every epoch-> 2860, tf.data.experimental.cardinality(train_batch)
# Higher batch sizes may be possible with a bigger instance. Current instance size is n1-standard-2.
BATCHES = 16


VOCAB_SIZE = 33000
# START_TOKEN and END_TOKEN indices were picked from the docstring_vocab.txt and code_vocab.txt.
# They share the same values as these are reserved tokens.
START_TOKEN = 2
END_TOKEN = 3

# Patience threshold for validation loss metric.
PATIENCE_THRESHOLD = 1

strategy = resolve_tpu_strategy(address=TPU_ADDRESS)
# Since, TPUs act distributed in nature. Therefore there are replicas of the model and the training is sped up by the
# number of replicas.
REPLICAS = strategy.num_replicas_in_sync
AUTO = tf.data.experimental.AUTOTUNE

tokenizer_params = {'lower_case': True}
code_tokenizer = text.BertTokenizer(BUCKET + 'code_vocab.txt', **tokenizer_params)
docstring_tokenizer = text.BertTokenizer(BUCKET + 'docstring_vocab.txt', **tokenizer_params)

print('Tokenizer loaded')

ensure_dataset_is_downloaded(file_name)
df = pd.read_pickle(file_name)
df_train, df_valid, df_test = randomize_dataset(df)


def tokenize(code, doc, code_tokenizer: text.BertTokenizer, docstring_tokenizer: text.BertTokenizer):
    """
    Part of the text pre-processing pipeline to convert string text to trimmed and padded sequences.
    The data may be provided in batches.

    :param code: Code string. (tf.RaggedTensor)
    :param doc: Document string. (tf.RaggedTensor)
    :param code_tokenizer: Custom text tokenizer for Python code.
    :param docstring_tokenizer: Custom text tokenizer for Python docstrings.
    :return: A tuple of tf.Tensor
    """
    code: tf.RaggedTensor = code_tokenizer.tokenize(code).merge_dims(-2, -1)
    doc: tf.RaggedTensor = docstring_tokenizer.tokenize(doc).merge_dims(-2, -1)

    # 2 tokens are to be added at the end and start of the tokenized input.
    trimmer = text.WaterfallTrimmer(MAX_INPUT_LEN - 2)
    code = trimmer.trim([code])
    doc = trimmer.trim([doc])

    # Padding Start and End token for each batch.
    code, _ = text.combine_segments(segments=code, start_of_sequence_id=START_TOKEN, end_of_segment_id=END_TOKEN)
    doc, _ = text.combine_segments(segments=doc, start_of_sequence_id=START_TOKEN, end_of_segment_id=END_TOKEN)

    # Embedding inputs are being padded so that a fixed input length vectors is fed to the model.
    code, _ = text.pad_model_inputs(input=code, max_seq_length=MAX_INPUT_LEN)
    doc, _ = text.pad_model_inputs(input=doc, max_seq_length=MAX_INPUT_LEN)

    return code, doc


training_dataset = Dataset.from_tensor_slices((df_train['code'].values, df_train['docstring'].values),
                                              name='Train_Dataset')
validation_dataset = Dataset.from_tensor_slices((df_valid['code'].values, df_valid['docstring'].values),
                                                name='Validation_Dataset')
test_dataset = Dataset.from_tensor_slices((df_test['code'].values, df_test['docstring'].values), name='Testing_Dataset')

print('Dataset loaded')

train_batch = training_dataset.prefetch(AUTO).batch(REPLICAS * BATCHES) \
    .map(lambda code, doc: tokenize(code, doc, code_tokenizer, docstring_tokenizer),
         num_parallel_calls=tf.data.AUTOTUNE).cache()

valid_batch = validation_dataset.prefetch(AUTO).batch(REPLICAS * BATCHES) \
    .map(lambda code, doc: tokenize(code, doc, code_tokenizer, docstring_tokenizer),
         num_parallel_calls=tf.data.AUTOTUNE).cache()

test_batch = test_dataset.prefetch(AUTO).batch(REPLICAS * BATCHES) \
    .map(lambda code, doc: tokenize(code, doc, code_tokenizer, docstring_tokenizer),
         num_parallel_calls=tf.data.AUTOTUNE).cache()

# All batches are made ready for distributions to each replica of the model in the TPU.
train_batch: tf.distribute.DistributedDataset = strategy.experimental_distribute_dataset(train_batch)
valid_batch: tf.distribute.DistributedDataset = strategy.experimental_distribute_dataset(valid_batch)
test_batch: tf.distribute.DistributedDataset = strategy.experimental_distribute_dataset(test_batch)

print('Distributed Batches ready')

learning_rate = Scheduler(REPLICAS * dim)

# Loading the model into each replica of the TPU.
with strategy.scope():
    optimizer = Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

    categorical_loss = SparseCategoricalCrossentropy(from_logits=True, reduction='none')

    train_loss = Mean(name='Training Loss')
    train_acc = Mean(name='Training Accuracy')

    validation_loss = Mean(name='Validation Loss')
    validation_acc = Mean(name='Validation Accuracy')

    test_loss = Mean(name='Test Loss')
    test_acc = Mean(name='Test Accuracy')

    model = TransformerModel(num_layers, dim=dim, heads=heads, inner_dim=inner_dim, seq_len=MAX_INPUT_LEN,
                             enc_vocab_size=VOCAB_SIZE, dec_vocab_size=VOCAB_SIZE, pe_enc_size=MAX_INPUT_LEN,
                             pe_dec_size=MAX_INPUT_LEN, dropout_rate=dropout_rate)

    # First input shape is for the input embeddings and the second is for the target embeddings.
    model.build(input_shape=[(None, MAX_INPUT_LEN), (None, MAX_INPUT_LEN - 1)])
    model.summary()

    checkpoint = tf.train.Checkpoint(transformer=model, optimizer=optimizer)
    checkpoint_manager = tf.train.CheckpointManager(checkpoint, checkpoint_path, max_to_keep=5)

    # Loading model's trainable variables if they are available via the checkpoint path.
    if checkpoint_manager.latest_checkpoint:
        result = checkpoint.restore(checkpoint_manager.latest_checkpoint).expect_partial()
        result.assert_existing_objects_matched()
        print(f'Model restored from checkpoint {result}')

print('Model loaded in scope')


def loss_function(y_real, y_pred):
    mask = tf.logical_not(y_real == 0)
    loss = categorical_loss(y_real, y_pred)

    mask = tf.cast(mask, dtype=loss.dtype)
    true_loss = loss * mask
    # Sum up loss across batches and devices.
    true_loss = tf.reduce_sum(true_loss) / tf.reduce_sum(mask)

    return true_loss


def accuracy_function(y_real, y_pred):
    dtype = tf.float32
    mask = tf.logical_not(y_real == 0)
    # Choose the vector from the embedding dimension with the highest probability.
    acc = (y_real == tf.argmax(y_pred, axis=2))
    acc = tf.cast(tf.logical_and(acc, mask), dtype=dtype)
    mask = tf.cast(mask, dtype=dtype)

    return tf.reduce_sum(acc) / tf.reduce_sum(mask)


def train_step(inputs):
    code, doc_string = inputs
    doc_string_inputs = doc_string[:, :-1]
    doc_string_targets = doc_string[:, 1:]

    with tf.GradientTape() as tape:
        doc_string_predictions = model([code, doc_string_inputs], training=True)
        loss_value = loss_function(doc_string_targets, doc_string_predictions)

    grads = tape.gradient(loss_value, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    acc = accuracy_function(doc_string_targets, doc_string_predictions)

    train_acc(acc)
    train_loss(loss_value)


def validate_step(inputs):
    code, doc_string = inputs
    doc_string_inputs = doc_string[:, :-1]
    doc_string_targets = doc_string[:, 1:]

    doc_string_predictions = model([code, doc_string_inputs], training=False)
    loss_value = loss_function(doc_string_targets, doc_string_predictions)

    acc = accuracy_function(doc_string_targets, doc_string_predictions)

    validation_acc(acc)
    validation_loss(loss_value)


def test_step(inputs):
    code = inputs[0]
    doc_string = inputs[1]
    doc_string_inputs = doc_string[:, :-1]
    doc_string_targets = doc_string[:, 1:]

    doc_string_predictions = model([code, doc_string_inputs], training=False)
    loss_value = loss_function(doc_string_targets, doc_string_predictions)
    acc = accuracy_function(doc_string_targets, doc_string_predictions)

    test_acc(acc)
    test_loss(loss_value)


# Stratergy run reference https://www.tensorflow.org/api_docs/python/tf/distribute/TPUStrategy#run
# https://www.tensorflow.org/guide/distributed_training#use_tfdistributestrategy_with_custom_training_loops

@tf.function
def tpu_training_step(inputs):
    strategy.run(train_step, args=(inputs,))


@tf.function
def tpu_validation_step(inputs):
    strategy.run(validate_step, args=(inputs,))


@tf.function
def tpu_test_step(inputs):
    strategy.run(test_step, args=(inputs,))


def train_model():
    previous_val_loss = None
    patience = 0
    history = []

    for epoch in range(EPOCHS):
        start_time = time()

        train_acc.reset_states()
        train_loss.reset_states()
        validation_acc.reset_states()
        validation_loss.reset_states()

        for batch, inputs in enumerate(train_batch):

            log = batch % 50 == 0
            batch_start_time = time() if log else None
            tpu_training_step(inputs)

            if log:
                print(f'Epoch {epoch + 1} Batch {batch} Training Accuracy {train_acc.result():.6f} '
                      f'Training Loss {train_loss.result():.6f} Time taken for 1 batch is '
                      f'{(time() - batch_start_time):.2f} seconds')

        for batch, inputs in enumerate(valid_batch):
            tpu_validation_step(inputs)

        current_val_loss = round(validation_loss.result(), 6)

        history.append((epoch, train_acc.result(), train_loss.result(), validation_acc.result(),
                        validation_loss.result()))

        # Early Stopping with custom validation loss patience.
        if previous_val_loss is not None and current_val_loss >= previous_val_loss:
            if patience == PATIENCE_THRESHOLD:
                print('Early stopping as model has converged.')
                checkpoint_result = checkpoint_manager.save()
                print(f'Saving checkpoint for epoch {epoch + 1} at {checkpoint_result}')
                print('*' * 64)
                break
            else:
                patience += 1
        else:
            previous_val_loss = current_val_loss
            patience = 0

        # Save checkpoint every 2 epochs.
        if (epoch + 1) % 2 == 0:
            checkpoint_result = checkpoint_manager.save()
            print(f'Saving checkpoint for epoch {epoch + 1} at {checkpoint_result}')

        print('*****')

        print(f'Epoch {epoch + 1} Accuracy {train_acc.result():.6f} Loss {train_loss.result():.6f}')
        print(f'Epoch {epoch + 1} Validation Accuracy {validation_acc.result():.6f} '
              f'Validation Loss {validation_loss.result():.6f}')

        print('*****')
        print(f'Time taken {(time() - start_time):.2f} seconds')
        print('*****')
        print('*****')

    print('End of epoch. Stopping training.')
    print('*' * 64)

    history_df = pd.DataFrame(history, columns=['Epochs', 'Training Accuracy', 'Training Loss', 'Validation Accuracy',
                                                'Validation Loss'])

    history_df.to_csv('./history_df.csv')

    export_csv(history_df.to_csv(), file_name=f'{model_history_path[len(BUCKET):]}history_df.csv')
    print('Model saved to bucket path')


def test_model():
    for batch, inputs in enumerate(test_batch):
        tpu_test_step(inputs)

    print(f'Test accuracy {test_acc.result()}')
    print(f'Test loss {test_loss.result()}')


if __name__ == '__main__':
    should_train_model, should_test_model = False, False
    while not (should_train_model or should_test_model):
        inputs = input('Would you like to train the model or perform testing on it?\n')
        should_train_model = 'train' in inputs.lower()
        should_test_model = 'test' in inputs.lower()

        if should_train_model or should_test_model:
            break
        else:
            print('Please enter a valid answer\n')

    if should_train_model:
        train_model()
        should_test_model = True
    if should_test_model:
        print('Testing the model')
        test_model()
