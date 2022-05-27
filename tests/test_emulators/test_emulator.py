#!/usr/bin/env python

import shutil

from olympus.datasets import Dataset, datasets_list
from olympus.emulators import Emulator, list_trained_emulators
from olympus.models import BayesNeuralNet, NeuralNet, get_models_list

EMULATED_DATASETS = [
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
    #
    "agnp",
    "autoam",
    "crossed_barrel",
    "p3ht",
    "thin_film",
    #
    "suzuki_i",
    "suzuki_ii",
    "suzuki_iii",
    "suzuki_iv",
]


def test_init_existing():
    emulator = Emulator(dataset="hplc", model="BayesNeuralNet")
    assert emulator.dataset.kind == "hplc"
    assert emulator.model.kind == "BayesNeuralNet"


def test_init_with_nones():
    emulator_0 = Emulator(dataset=None, model="BayesNeuralNet")
    emulator_1 = Emulator(dataset="hplc", model=None)


def test_init_with_objects():
    dataset = Dataset(kind="hplc", test_frac=0.3)
    model = BayesNeuralNet(max_epochs=1000)
    _ = Emulator(
        dataset=dataset,
        model=model,
        feature_transform="standardize",
        target_transform="mean",
    )


def test_loading_all_emulators():
    models_list = get_models_list()
    for dataset in EMULATED_DATASETS:
        # for model in ['BayesNeuralNet', 'NeuralNet']:
        for model in ["BayesNeuralNet"]:
            _ = Emulator(dataset=dataset, model=model)


def test_emulation():
    emulator = Emulator(dataset="hplc", model="BayesNeuralNet")
    values = emulator.run([0, 0, 0, 0, 0, 0], return_paramvector=True)
    # values is a list of ParamVectors. Each ParamVector for a sample. We have only one sample unless we probe
    # batches of parameters. A ParamVector can then have multiple keys if we have multiple objectives.
    assert len(values) == 1  # we expect a list with only 1  ParamVector
    assert len(values[0].to_array()) == 1  # we expect only 1 objective
    value = values[0].to_array()[0]
    assert isinstance(value, float)
    assert value >= 0


def test_train_bnn():
    # only tests if the code
    dataset = Dataset(kind="hplc", num_folds=5)
    model = BayesNeuralNet(scope="hplc", max_epochs=2)
    emulator = Emulator(dataset=dataset, model=model)
    emulator.train()
    emulator.save("emulator_test")
    shutil.rmtree("emulator_test")


def test_train_nn():
    # only tests if the code
    dataset = Dataset(kind="hplc", num_folds=5)
    model = NeuralNet(scope="hplc", max_epochs=2)
    emulator = Emulator(dataset=dataset, model=model)
    emulator.train()
    emulator.save("emulator_test")
    shutil.rmtree("emulator_test")


def test_cross_validate():
    # only tests if the code runs
    dataset = Dataset(kind="hplc", num_folds=5)
    model = BayesNeuralNet(scope="hplc", max_epochs=2)
    emulator = Emulator(dataset=dataset, model=model)
    emulator.cross_validate()
    emulator.save("emulator_test")
    shutil.rmtree("emulator_test")


def test_list_emulators():
    # only tests if the code runs
    list_trained_emulators()


# ======================================


if __name__ == "__main__":
    test_train_nn()
