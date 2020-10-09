#!/usr/bin/env python

import numpy as np

from olympus.datasets import Dataset
from olympus.utils.data_transformer import DataTransformer


np.random.seed(100691)
data = np.random.uniform(low=0, high=1, size=(3, 2))


def test_train_identity_array():
    data_transformer = DataTransformer(transformations="identity")
    data_transformer.train(data)

    assert np.all(data_transformer._min == np.amin(data, axis=0))
    assert np.all(data_transformer._max == np.amax(data, axis=0))
    assert np.all(data_transformer._stddev == np.std(data, axis=0))
    assert np.all(data_transformer._mean == np.mean(data, axis=0))


def test_train_standardize_array():
    data_transformer = DataTransformer(transformations="standardize")
    data_transformer.train(data)
    transformed = data_transformer.transform(data)
    assert np.all(np.abs(np.mean(transformed, axis=0)) < 1e-7)
    assert np.all(np.abs(np.std(transformed, axis=0) - 1.0) < 1e-7)
    assert np.all(data == data_transformer.back_transform(transformed))


def test_train_normalize_array():
    data_transformer = DataTransformer(transformations="normalize")
    data_transformer.train(data)
    transformed = data_transformer.transform(data)
    assert np.all(np.abs(np.amin(transformed, axis=0)) < 1e-7)
    assert np.all(np.abs(np.amax(transformed, axis=0) - 1.0) < 1e-7)
    assert np.all(data == data_transformer.back_transform(transformed))


def test_train_mean_array():
    data_transformer = DataTransformer(transformations="mean")
    data_transformer.train(data)
    transformed = data_transformer.transform(data)
    assert np.all(np.abs(np.mean(transformed, axis=0) - 1.0) < 1e-7)
    assert np.all(data == data_transformer.back_transform(transformed))


def test_train_identity_array():
    data_transformer = DataTransformer(transformations="identity")
    data_transformer.train(data)
    transformed = data_transformer.transform(data)
    assert np.all(data == transformed)
    assert np.all(data == data_transformer.back_transform(transformed))


def test_train_log_mean_array():
    data_transformer = DataTransformer(transformations="log_mean")
    data_transformer.train(data)
    transformed = data_transformer.transform(data)
    assert np.all(data == data_transformer.back_transform(transformed))


def test_train_sqrt_mean_array():
    data_transformer = DataTransformer(transformations="sqrt_mean")
    data_transformer.train(data)
    transformed = data_transformer.transform(data)
    assert np.sum(np.abs(data - data_transformer.back_transform(transformed))) < 1e-7


# def test_train_periodic_array():
# 	data = Dataset('excitonics')
# 	data_transformer = DataTransformer(transformations='periodic')
# 	data_transformer.train(data)
# 	transformed = data_transformer.transform(data)
# 	assert data.shape[1] == 11
# 	assert transformed.shape[1] == 15


if __name__ == "__main__":
    test_train_periodic_array()
