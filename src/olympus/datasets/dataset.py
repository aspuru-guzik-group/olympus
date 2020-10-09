#!/usr/bin/env python

import json
from pandas import DataFrame, read_csv
import numpy as np

from olympus import Logger
from olympus.campaigns import ParameterSpace
from olympus.objects import Parameter

import os
from glob import glob


# =========================
# Main Class of This Module
# =========================
class Dataset:
    """
    A ``Dataset`` object stores the data of a dataset by wrapping a ``pandas.DataFrame`` in its ``data`` attribute, provides
    additional information on the dataset, and provides convenience methods to access features and targets as well as
    to generate training/validation/test splits.

    Args:
        kind (str): kind of the Olympus dataset to load.
        data (array): custom dataset. Same input as for pandas.DataFrame.
        columns (list): column names. Same input as for pandas.DataFrame.
        target_ids (list): list of column indices, or names if provided, that identify the targets for the predictions.
        test_frac (float): fraction of the data to be used as test set.
        num_folds (int): number of cross validation folds the training set will be split into.
        random_seed (int): random seed for numpy. Setting a seed makes the random splits reproducible.
    """

    def __init__(
        self,
        kind=None,
        data=None,
        columns=None,
        target_ids=None,
        test_frac=0.2,
        num_folds=5,
        random_seed=None,
    ):

        _validate_dataset_args(kind, data, columns, target_ids)

        self.kind = kind
        self.name = kind
        self.test_frac = test_frac
        self.num_folds = num_folds
        self.random_seed = random_seed

        # we defined the param_space as part of the Dataset object. When we
        # load a standard dataset, we define the param_space from the config
        # file. If Dataset is a custom dataset, the user will need to set_param_space
        self.param_space = None
        self.value_space = None

        # ------------------------------------------
        # Case 1: data directly provided by the user
        # ------------------------------------------
        if kind is None:
            self._description = "Custom Dataset"
            if target_ids is None:
                self._targets = []
            else:
                self._targets = target_ids
            _data = data

        # ------------------------------
        # Case 2: Olympus dataset loaded
        # ------------------------------
        elif kind is not None:
            _data, _config, self._description = load_dataset(kind)
            self._targets = [t["name"] for t in _config["measurements"]]
            columns = [f["name"] for f in _config["parameters"]] + self._targets
            # create param_space from config file
            self._create_param_space(_config)
            self._create_value_space(_config)
            # define attributes of interest - done here so to avoid calling load_dataset again
            self.constraints = _config["constraints"]
            self._goal = _config["default_goal"]

        # the numeric data is stored here, wrapping a DataFrame
        self.data = DataFrame(data=_data, index=None, columns=columns)

        # make sure the targets are the last column(s)
        for tgt in self._targets:
            assert tgt in list(self.data.columns)[-len(self._targets) :]

        # create dataset splits
        self.create_train_validate_test_splits()

    @property
    def goal(self):
        if not "_goal" in self.__dict__:
            _data, _config, _description = load_dataset(self.kind)
            self._goal = _config["default_goal"]
        return self._goal

    @property
    def measurement_name(self):
        if not "_measurement_name" in self.__dict__:
            _data, _config, _description = load_dataset(self.kind)
            self._measurement_name = _config["measurements"][0]["name"]
        return self._measurement_name

    def __getattr__(self, attr):
        if attr == "__getstate__":
            return lambda: self.__dict__
        elif attr == "__setstate__":

            def set_state(_dict):
                self.__dict__ = _dict

            return set_state
        elif attr == "kind" and not attr in self.__dict__:
            val = getattr(self, "name")
            setattr(self, attr, val)
            return val
        print('ATTR', attr)
        return getattr(self, attr)

    def __str__(self):
        # TODO: add some more info here...
        string = f"Dataset(kind={self.kind})"
        return string

    @property
    def size(self):
        return self.data.shape[0]

    @property
    def shape(self):
        return self.data.shape

    # we could also overwrite the DataFrame.info method
    def dataset_info(self):
        """Provide summary info about dataset.
        """
        Logger.log(self._description, "INFO")

    def set_param_space(self, param_space):
        """Define the parameter space of the dataset.

        Args:
            param_space (ParameterSpace): ParameterSpace object with information about all variables in the dataset.

        """

        # validate first...
        for i, param in enumerate(param_space.parameters):
            # check names are matching
            if self.feature_names[i] != param.name:
                message = f"Parameter name `{self.feature_names[i]}` does not match name `{param.name}` found in dataset!"
                Logger.log(message, "WARNING")
            # check data provided is within param_space bounds
            if param.low > np.min(self.data.iloc[:, i]):
                message = f"Lower bound of {param.low} provided for parameter `{param.name}` is higher than minimum found in the data!"
                Logger.log(message, "ERROR")
            if param.high < np.min(self.data.iloc[:, i]):
                message = f"Upper bound of {param.high} provided for parameter `{param.name}` is lower than maximum found in the data!"
                Logger.log(message, "ERROR")

        # ...then assign
        self.param_space = param_space

    def infer_param_space(self):
        """Guess the parameter space from the dataset. The range for all parameters will be define based on the
        minimum and maximum values in the dataset for each variable. All variables will be assumed not to be periodic.
        """
        param_space = ParameterSpace()
        for name in self.feature_names:
            param_dict = {
                "name": name,
                "low": np.min(self.data.loc[:, name]),
                "high": np.max(self.data.loc[:, name]),
            }
            param = Parameter().from_dict(param_dict)
            param_space.add(param)

        # set the param space
        self.set_param_space(param_space)

    def to_disk(self, folder='custom_dataset'):
        """Save the dataset to disk in the format expected by Olympus for its own datasets. This can be useful if you
        plan to upload the dataset to the community datasets available online.

        Args:
            folder (str): Folder in which to save the dataset files.
        """

        # create the folder
        os.mkdir(folder)

        # save the numeric data
        self.data.to_csv(f'{folder}/data.csv', header=False, index=False)

        # save the description
        self._generate_description()
        with open(f'{folder}/description.txt', 'w') as f:
            f.write(self._description)

        # save the config file
        _config = {}
        _config['constraints'] = {'parameters': 'none', 'measurements': 'none'}

        _config['parameters'] = []
        for param in self.param_space.parameters:
            d = {'name': param.name, 'type': param.kind, 'lower': param.low, 'upper':param.high}
            _config['parameters'].append(d)

        _config['measurements'] = []
        for target in self._targets:
            t = {'name': target, 'type': 'continuous'}
            _config['measurements'].append(t)

        with open(f'{folder}/config.json', 'w') as f:
            f.write(json.dumps(_config, indent=4, sort_keys=True))

    def _generate_description(self):
        _description = []
        if self.kind is None:
            _description.append('Custom Dataset\n')
        else:
            _description.append(f'{self.kind}\n')

        _description.append('=========================================')
        _description.append('                Summary')
        _description.append('-----------------------------------------')
        _description.append(f'    Number of Samples       {self.size:>10}')
        _description.append(f'    Dimensionality          {len(self.feature_names):>10}')
        _description.append(f'    Features:')
        for param in self.param_space.parameters:
            _description.append(f'        {param.name:<10}          {param.kind:>10}')
        _description.append(f'    Targets:')
        for target in self._targets:
            _description.append(f'        {target:<10}          continuous')

        self._description = "\n".join(_description)

    # ----------------------------------
    # Methods about features and targets
    # ----------------------------------
    @property
    def feature_names(self):
        # return names of features
        return [c for c in self.data.columns if c not in self._targets]

    @property
    def target_names(self):
        # return names of targets
        return self._targets

    @property
    def features(self):
        # return data for features
        return self.data.loc[:, self.feature_names]

    @property
    def targets(self):
        # return data for targets
        if type(self.target_names[0]) == str:
            return self.data.loc[:, self.target_names]
        elif type(self.target_names[0]) == int:
            return self.data.iloc[:, self.target_names]

    @property
    def features_dim(self):
        return len(self.feature_names)

    @property
    def targets_dim(self):
        return len(self.target_names)

    # ----------------------------------------------------------------
    # Methods for dataset splitting into training/validation/test sets
    # ----------------------------------------------------------------
    def create_train_validate_test_splits(
        self, test_frac=0.2, num_folds=5, test_indices=None
    ):
        """
        Args:
            test_frac (float)
            num_folds (int)
            test_indices (array)
                Array with the indices of the samples to be used as test set.
        """
        # update num_folds and test frac
        self.test_frac = test_frac
        if test_indices is not None:
            self.test_frac = len(test_indices) / len(self.data)
        self.num_folds = num_folds

        np.random.seed(self.random_seed)  # set random seed
        nrows = self.data.shape[0]
        indices = range(nrows)
        n_train_samples = int(round(nrows * self.test_frac, ndigits=0))

        # test set
        if test_indices is None:
            self.test_indices = np.random.choice(
                indices, size=n_train_samples, replace=False
            )
        else:
            self.test_indices = np.array(test_indices)

        # training set
        self.train_indices = np.setdiff1d(indices, self.test_indices)

        # cross-validation sets are subsets of the training set
        train_indices_shuffled = np.random.permutation(
            self.train_indices
        )  # random shuffle
        self.cv_fold_indices = np.array_split(
            train_indices_shuffled, self.num_folds
        )  # split in approx equal sized arrays
        # define train/valid indices for each fold
        self.cross_val_indices = []
        for i in range(self.num_folds):
            fold_train = np.concatenate(
                np.delete(self.cv_fold_indices, i, axis=0), axis=0
            )
            fold_valid = self.cv_fold_indices[i]
            self.cross_val_indices.append([fold_train, fold_valid])
        # TODO: unfix random seed?

    @property
    def train_set(self):
        # The training set with features+targets
        return self.data.loc[self.train_indices, :]

    @property
    def train_set_features(self):
        # The training set with only the features
        return self.data.loc[self.train_indices, self.feature_names]

    @property
    def train_set_targets(self):
        # The training set with only the targets
        return self.data.loc[self.train_indices, self.target_names]

    @property
    def test_set(self):
        # The test set with features+targets
        return self.data.loc[self.test_indices, :]

    @property
    def test_set_features(self):
        # The test set with only the features
        return self.data.loc[self.test_indices, self.feature_names]

    @property
    def test_set_targets(self):
        # The test set with only the targets
        return self.data.loc[self.test_indices, self.target_names]

    @property
    def cross_val_sets(self):
        # List of cv sets with features+targets
        cv_sets = []
        for train_id, valid_id in self.cross_val_indices:
            train_data = self.data.loc[train_id, :]
            valid_data = self.data.loc[valid_id, :]
            cv_sets.append((train_data, valid_data))
        return cv_sets

    @property
    def cross_val_sets_features(self):
        # List of cv sets with only features
        cv_sets = []
        for train_id, valid_id in self.cross_val_indices:
            train_data = self.data.loc[train_id, self.feature_names]
            valid_data = self.data.loc[valid_id, self.feature_names]
            cv_sets.append((train_data, valid_data))
        return cv_sets

    @property
    def cross_val_sets_targets(self):
        # List of cv sets with only targets
        cv_sets = []
        for train_id, valid_id in self.cross_val_indices:
            train_data = self.data.loc[train_id, self.target_names]
            valid_data = self.data.loc[valid_id, self.target_names]
            cv_sets.append((train_data, valid_data))
        return cv_sets

    def get_cv_fold(self, fold):
        """Get the data for a specific cross-validation fold.

        Args:
            fold (int): fold id.

        Returns:
            data (DataFrame): data for the chosen fold.
        """
        indices = self.cv_fold_indices[fold]
        return self.data.loc[indices, :]

    # ---------------
    # Private Methods
    # ---------------
    def _create_param_space(self, config):
        self.param_space = ParameterSpace()
        self.param_space.add(
            [Parameter().from_dict(feature) for feature in config["parameters"]]
        )

    def _create_value_space(self, config):
        self.value_space = ParameterSpace()
        self.value_space.add(
            [Parameter().from_dict(feature) for feature in config["measurements"]]
        )


# ===============
# Other Functions
# ===============
def load_dataset(kind):
    """Loads a dataset from the Olympus dataset library.

    Args:
        kind (str): kind of Olympus dataset to load.

    Returns:
        (tuple): tuple containing:
            data (array): numpy array where each row is a sample and each column is a feature/target. The columns are
            sorted such features come first and then targets.
            config (dict): dict containing information on the features and targets present in data.
            description (str): string describing the dataset.
    """

    _validate_dataset_args(kind=kind, data=None, columns=None, target_names=None)
    datasets_path = os.path.dirname(os.path.abspath(__file__))

    # load description
    with open("".join(f"{datasets_path}/dataset_{kind}/description.txt")) as txtfile:
        description = txtfile.read()

    # load info on features/targets
    with open("".join(f"{datasets_path}/dataset_{kind}/config.json"), "r") as content:
        config = json.loads(content.read())

    # load data
    csv_file = "".join(f"{datasets_path}/dataset_{kind}/data.csv")
    try:
        data = read_csv(csv_file, header=None).to_numpy()
    except FileNotFoundError:
        Logger.log(f"Could not find data.csv for dataset {kind}", "FATAL")

    return data, config, description


def _validate_dataset_args(kind, data, columns, target_names):
    if kind is not None:
        # -----------------------------------
        # check that a correct name is passed
        # -----------------------------------
        # TODO: reduce redundant code by importing the list from where we have it already
        module_path = os.path.dirname(os.path.abspath(__file__))
        olympus_datasets = []
        for dir_name in glob(f"{module_path}/dataset_*"):
            dir_name = dir_name.split("/")[-1][8:]
            olympus_datasets.append(dir_name)
        if kind not in olympus_datasets:
            message = (
                "Could not find dataset `{0}`. Please choose from one of the available "
                "datasets: {1}.".format(kind, ", ".join(list(olympus_datasets)))
            )
            Logger.log(message, "FATAL")
        # --------------------------------------------------------------
        # we will discard these arguments, so check if they are provided
        # --------------------------------------------------------------
        if data is not None:
            message = (
                "One of the Olympus datasets has been loaded via the argument `kind`, argument `data` "
                "will be discarded"
            )
            Logger.log(message, "WARNING")
        if columns is not None:
            message = (
                "One of the Olympus datasets has been loaded via the argument `kind`, argument `columns` "
                "will be discarded"
            )
            Logger.log(message, "WARNING")
        if target_names is not None:
            message = (
                "One of the Olympus datasets has been loaded via the argument `kind`, argument "
                "`target_names` will be discarded"
            )
            Logger.log(message, "WARNING")
