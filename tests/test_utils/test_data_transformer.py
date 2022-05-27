#!/usr/bin/env python

import numpy as np

from olympus.datasets import Dataset
from olympus.utils.data_transformer import (
    DataTransformer,
    simpl_to_cube,
    cube_to_simpl,
)


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


def test_simpl_to_cube():
    simpls = np.array([[0.1, 0.3, 0.4, 0.2]])
    cubes = simpl_to_cube(simpls)
    assert cubes.shape[1] == simpls.shape[1]-1
    assert cubes.shape[0] == 1

def test_cube_to_simpl():
    cubes = np.array([[0.8324, 0.1903, 0.6787]])
    simpls = cube_to_simpl(cubes)
    assert simpls.shape[1] == cubes.shape[1]+1
    assert simpls.shape[0] == 1
    assert np.isclose(np.sum(simpls), 1.,)



if __name__ == "__main__":
    test_train_periodic_array()
