import os
import numpy as np
import tensorflow as tf


def ensure_dataset_is_downloaded(file_name):
    """
    Ensures that the dataset is available on the GCloud Compute Engine. The dataset is not with the TPU.

    :param file_name: Pickle file's name
    """
    if not os.path.isfile(os.getcwd() + f'/{file_name}'):
        # download dataset using gcloud
        from google.cloud import storage
        BUCKET_NAME = 'code-search-python-dataset'
        creds = '/gcs_key.json'
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.getcwd() + creds

        client = storage.Client()
        dataset_bucket = client.get_bucket(BUCKET_NAME)
        blob = dataset_bucket.get_blob(file_name)

        print('Downloading dataset from GCS')

        with open(file_name, mode='wb') as f:
            blob.download_to_file(f)

        del os.environ["GOOGLE_APPLICATION_CREDENTIALS"]

        print('File Downloaded')
    else:
        print('Dataset exists')


def randomize_dataset(df, split_size=0.8, seed=42):
    """
    A seeded version of train, validation and split.

    :param df: Dataframe to be split.
    :param split_size: Portion of the dataset for training. The rest is equally divided between validation and test set.
    :param seed: Seed value for numpy.
    :return: A tuple of 3 Dataframes representing train, validation and test.
    """
    np.random.seed(seed)
    mask_train = np.random.rand(len(df)) < split_size

    df_train = df[mask_train].reset_index(drop=True)
    df_test = df[~mask_train].reset_index(drop=True)

    mask_test = np.random.rand(len(df_test)) < 0.5

    df_valid = df_test[~mask_test].reset_index(drop=True)
    df_test = df_test[mask_test].reset_index(drop=True)

    return df_train, df_valid, df_test


def resolve_tpu_strategy(address) -> tf.distribute.TPUStrategy:
    """
    Resolves the TPU Cluster and initializes the TPU. And then returns the strategy.

    :param address: TPU Node's name.
    :return: TPUStrategy object.
    """
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu=address)
    tf.config.experimental_connect_to_cluster(resolver)
    tf.tpu.experimental.initialize_tpu_system(resolver)
    print("All devices: ", tf.config.list_logical_devices('TPU'))

    strategy = tf.distribute.TPUStrategy(resolver)

    return strategy


def export_csv(df_str, file_name):
    """
    Exports string Dataframe as a csv file.

    :param df_str: string representation of a Dataframe.
    :param file_name: bucket file path.
    """
    from google.cloud import storage
    BUCKET_NAME = 'code-search-python-dataset'
    creds = '/gcs_key.json'
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.getcwd() + creds

    client = storage.Client()
    dataset_bucket = client.get_bucket(BUCKET_NAME)
    dataset_bucket.blob(file_name).upload_from_string(df_str, content_type='text/csv')

    del os.environ["GOOGLE_APPLICATION_CREDENTIALS"]


def ensure_model_checkpoint_is_downloaded(local_file_path, checkpoint_path):
    if not os.path.isdir(local_file_path):
        from google.cloud import storage
        BUCKET_NAME = 'code-search-python-dataset'
        creds = '/gcs_key.json'
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.getcwd() + creds

        client = storage.Client()
        dataset_bucket = client.get_bucket(BUCKET_NAME)
        blob_list = dataset_bucket.list_blobs(prefix=checkpoint_path)
        os.mkdir(local_file_path)
        for blob in blob_list:
            blob_name = blob.name
            if blob_name[-1] != '/':
                file_name = blob_name.split('/')[-1]
                with open(local_file_path + f'/{file_name}', mode='wb') as f:
                    blob.download_to_file(f)

        del os.environ["GOOGLE_APPLICATION_CREDENTIALS"]
        print('Checkpoints downloaded')
    else:
        print('Checkpoints already downloaded')
