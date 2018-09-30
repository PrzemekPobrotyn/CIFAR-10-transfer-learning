import math
import os
import pickle
import tarfile
import time
import zipfile

import numpy as np
import requests
from tqdm import tqdm
from skimage.transform import resize


class Timer:
    """Context manager for time profiling."""
    def __enter__(self):
        self.start = time.time()

    def __exit__(self, type, value, traceback):
        print(f'Elapsed: {time.time() - self.start:.2f} seconds. \n')


def download_and_extract(
        url='https://www.cs.toronto.edu/%7Ekriz/cifar-10-python.tar.gz',
        target_dir=None,
):
    """Download and extract CIFAR-10"""
    target_dir = target_dir or os.getcwd()

    filename = url.split('/')[-1]
    r = requests.get(url, stream=True)

    # total size in bytes
    total_size = int(r.headers.get('content-length', 0))
    block_size = 1024

    with open(filename, 'wb') as file_handle:
        for data in tqdm(
                r.iter_content(block_size),
                total=math.ceil(total_size // block_size),
                unit='KB',
                unit_scale=True
        ):
            file_handle.write(data)

    # extract if necessary
    if filename.endswith(".zip"):
        with zipfile.ZipFile(filename, "r") as zip_handle:
            zip_handle.extractall(target_dir)
            # since data was extracted, remove the zip file
            os.remove(filename)
    elif filename.endswith((".tar.gz", ".tgz")):
        with tarfile.open(filename, "r:gz") as tar_handle:
            tar_handle.extractall(target_dir)
            # since data was extracted, remove the tar file
            os.remove(filename)


def unpickle(file):
    with open(file, 'rb') as fo:
        return pickle.load(fo, encoding='bytes')


def read_and_reshape_cifar10(data_dir='cifar-10-batches-py'):
    X_train, y_train = [], []
    X_val, y_val = [], []
    X_test, y_test = [], []

    for file in os.listdir(data_dir):
        if file.startswith('data') and file[-1] != '5':
            d = unpickle(os.path.join(data_dir, file))
            X_train.append(d[b'data'])
            y_train.append(d[b'labels'])
        elif file.startswith('data') and file[-1] == '5':
            d = unpickle(os.path.join(data_dir, file))
            X_val.append(d[b'data'])
            y_val.append(d[b'labels'])
        elif file == 'test_batch':
            d = unpickle(os.path.join(data_dir, file))
            X_test.append(d[b'data'])
            y_test.append(d[b'labels'])

    X_train = np.concatenate(X_train)
    y_train = np.concatenate(y_train)
    X_val = np.concatenate(X_val)
    y_val = np.concatenate(y_val)
    X_test = np.concatenate(X_test)
    y_test = np.concatenate(y_test)

    X_train = np.transpose(X_train.reshape(-1, 3, 32, 32), axes=(0, 2, 3, 1))
    X_val = np.transpose(X_val.reshape(-1, 3, 32, 32), axes=(0, 2, 3, 1))
    X_test = np.transpose(X_test.reshape(-1, 3, 32, 32), axes=(0, 2, 3, 1))

    d = unpickle('cifar-10-batches-py/batches.meta')
    label_names = [x.decode('utf-8') for x in d[b'label_names']]

    return X_train, y_train, X_val, y_val, X_test, y_test, label_names


def resize_images(batch_data, target_size, preserve_range=True):
    resized_batch = np.array(
        [resize(batch_data[i], target_size, preserve_range=preserve_range)
         for i in range(batch_data.shape[0])]
    ).astype('float32')
    return resized_batch
