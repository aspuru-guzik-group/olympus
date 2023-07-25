#!/usr/bin/env python

import pandas as pd
import pytest

from olympus.datasets import Dataset, datasets_list

FULL_CAT_DATASETS = ["perovskites", "dye_lasers", "redoxmers"]

DESC_DATASETS = ["perovskites", "redoxmers"]  # datasets which have descriptors

MIXED_DATASETS = ["suzuki_i", "suzuki_ii", "suzuki_iii", "suzuki_iv"]

FULL_CONT_DATASETS = [
    "snar",
    "photo_wf3",
    "benzylation",
    "fullerenes",
    "colors_bob",
    "photo_pce10",
    "alkox",
    "hplc",
    "colors_n9",
    "suzuki",
    "agnp",
    "autoam",
    "crossed_barrel",
    "p3ht",
    "thin_film",
]

MOO_DATASETS = [
    "dye_lasers",
    "redoxmers",
    "suzuki_i",
    "suzuki_ii",
    "suzuki_iii",
    "suzuki_iv",
]

SIMPLEX_CONSTRAINED_DATASETS = ["thin_film", "photo_pce10", "photo_wf3"]

ALL_DATASETS = datasets_list


@pytest.mark.parametrize(
    "test_frac, num_folds",
    [(0.2, 3), (0.2, 5), (0.2, 10), (0.3, 3), (0.3, 5), (0.3, 10)],
)
def test_dataset_init(test_frac, num_folds):
    for kind in ALL_DATASETS:
        _ = Dataset(kind=kind, test_frac=test_frac, num_folds=num_folds)


@pytest.mark.parametrize("kind", FULL_CAT_DATASETS)
def test_categorical(kind):
    dataset = Dataset(kind=kind)
    assert dataset.dataset_type == "full_cat"
    if kind in DESC_DATASETS:
        assert isinstance(dataset.descriptors, pd.DataFrame)
        assert dataset.descriptors.shape[0] > 0
    else:
        # empty DataFrame
        assert isinstance(dataset.descriptors, pd.DataFrame)
        assert dataset.descriptors.shape[0] == 0


@pytest.mark.parametrize("kind", MIXED_DATASETS)
def test_mixed(kind):
    dataset = Dataset(kind=kind)
    assert dataset.dataset_type == "mixed"


@pytest.mark.parametrize("kind", MOO_DATASETS)
def test_moo(kind):
    dataset = Dataset(kind=kind)
    assert len(dataset.value_space) > 1


@pytest.mark.parametrize("kind", SIMPLEX_CONSTRAINED_DATASETS)
def test_simplex_constrained_datasets(kind):
    dataset = Dataset(kind)
    assert dataset.parameter_constriants == "simplex"
    assert len(dataset.aux_param_space) == len(dataset.param_space) - 1

    for param_ix, param in enumerate(dataset.aux_param_space):
        assert param.name == f"param_{param_ix}"
        assert param.low == 0.0
        assert param.high == 1.0
