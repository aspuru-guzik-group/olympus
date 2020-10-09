#!/usr/bin/env python

import pytest
from olympus.datasets import datasets_list, Dataset


@pytest.mark.parametrize("test_frac, num_folds", [(0.2, 3), (0.2, 5), (0.2, 10), (0.3, 3), (0.3, 5), (0.3, 10)])
def test_dataset_init(test_frac, num_folds):
    for kind in datasets_list:
        _ = Dataset(kind=kind, test_frac=test_frac, num_folds=num_folds)
