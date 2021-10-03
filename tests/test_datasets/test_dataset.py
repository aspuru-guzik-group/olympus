#!/usr/bin/env python

import pytest
import pandas as pd
from olympus.datasets import datasets_list, Dataset


FULL_CAT_DATASETS = ['perovskites']
DESC_DATASETS = ['perovskites']
FULL_CONT_DATASETS = [
    'snar', 'photo_wf3', 'benzylation',
    'fullerenes', 'colors_bob', 'photo_pce10',
    'alkox', 'hplc', 'colors_n9', 'suzuki',
]
MIXED_DATASETS = []
ALL_DATASETS = datasets_list

@pytest.mark.parametrize(
    "test_frac, num_folds",
    [(0.2, 3), (0.2, 5), (0.2, 10), (0.3, 3), (0.3, 5), (0.3, 10)]
)
def test_dataset_init(test_frac, num_folds):
    for kind in ALL_DATASETS:
        _ = Dataset(kind=kind, test_frac=test_frac, num_folds=num_folds)


@pytest.mark.parametrize("kind", FULL_CAT_DATASETS)
def test_categorical(kind):
    dataset = Dataset(kind=kind)
    assert dataset.dataset_type == 'full_cat'
    if kind in DESC_DATASETS:
        assert isinstance(dataset.descriptors, pd.DataFrame)
    else:
        assert isinstance(dataset.descriptors, None)
