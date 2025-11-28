import pandas as pd
import numpy as np
import argparse
import random


def get_train_valid(X, y, valid_years=1):
    num_samples = X.shape[0]
    num_valid = valid_years * 365 * 24

    train_X = X[:num_samples-num_valid]
    train_y = y[:num_samples-num_valid]

    valid_X = X[num_samples-num_valid:]
    valid_y = y[num_samples-num_valid:]

    return train_X, train_y, valid_X, valid_y


def get_batches(X, y, batch_size, n_batches, input_seq_len, output_seq_len, input_dim):
    batches_x = np.zeros((n_batches, batch_size, input_seq_len, input_dim)).astype('float32')
    batches_y = np.zeros((n_batches, batch_size, output_seq_len)).astype('float32')

    checkpoint = X.shape[0] - input_seq_len - output_seq_len - 1


    for b in range(n_batches):
        sample_split_points = np.random.randint(input_seq_len, checkpoint, size=batch_size)
        rand_intervals = [(p - input_seq_len, p) for p in sample_split_points]
        batches_x[b, :, :] = np.array([X[p[0]:p[1]] for p in rand_intervals])
        batches_y[b, :] = np.array([y[p[1]:p[1] + output_seq_len] for p in rand_intervals])
    return batches_x, batches_y

