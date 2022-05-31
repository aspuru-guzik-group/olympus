#!/usr/bin/env python

import json
import os
from glob import glob

import numpy as np
from pandas import DataFrame, read_csv

from olympus import Logger
from olympus.campaigns.param_space import ParameterSpace
from olympus.objects import (
    Parameter,
    ParameterCategorical,
    ParameterContinuous,
    ParameterDiscrete,
    ParameterVector,
)

# To silence VisibleDeprecationWarning we use 'ignore'. We can use 'error' to get a traceback and resolve the issue.
np.warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

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
            _data, _config, self._description, _descriptors = load_dataset(
                kind
            )
            self._targets = [t["name"] for t in _config["measurements"]]
            columns = [
                f["name"] for f in _config["parameters"]
            ] + self._targets
            # descriptors are stored in the descriptors attribute
            desc_columns = ["param", "option", "name", "value"]
            self.descriptors = DataFrame(
                data=_descriptors, index=None, columns=desc_columns
            )
            # create param_space from config file
            self._create_param_space(_config)
            self._create_value_space(_config)

            # TODO: add full discrete to these types. Do we need mixed_cat_discrete, mixed_cont_cat etc..
            # level of specificity for the dataset types??
            # store the dataset type in an attribute ('full_cont', 'full_cat', 'mixed')
            # TODO: for now, full_cont could include continous or discrete parameters, should we change this?
            if np.all(
                [
                    param["type"] in ["categorical", "discrete"]
                    for param in self.param_space
                ]
            ):
                self.dataset_type = "full_cat"
            elif np.all(
                [param["type"] in ["continuous"] for param in self.param_space]
            ):
                self.dataset_type = "full_cont"
            else:
                self.dataset_type = "mixed"

            # param_type attribute stores unique parameter types for the dataset
            self.param_types = list(
                set([param["type"] for param in self.param_space])
            )

            # define attributes of interest - done here so to avoid calling load_dataset again
            self.constraints = _config["constraints"]

            # meta information for the evaluator
            self.parameter_constriants = self.constraints["parameters"]

            # if we have a "simplex" constrained parameter space, create an auxillary
            # parameter space of dimensions n-1 (n is the original feature dimension)
            if self.parameter_constriants == "simplex":
                self._create_aux_param_space(len(self.param_space))
            else:
                self.aux_param_space = ParameterSpace()

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
            _data, _config, _description, _ = load_dataset(self.kind)
            self._goal = _config["default_goal"]
        return self._goal

    @property
    def measurement_name(self):
        if not "_measurement_name" in self.__dict__:
            _data, _config, _description, _ = load_dataset(self.kind)
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
        print("ATTR", attr)
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
        """Provide summary info about dataset."""
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

            if param.type in ["discrete", "continuous"]:
                # check data provided is within param_space bounds
                if param.low > np.min(self.data.iloc[:, i]):
                    message = f"Lower bound of {param.low} provided for parameter `{param.name}` is higher than minimum found in the data!"
                    Logger.log(message, "ERROR")
                if param.high < np.min(self.data.iloc[:, i]):
                    message = f"Upper bound of {param.high} provided for parameter `{param.name}` is lower than maximum found in the data!"
                    Logger.log(message, "ERROR")
            elif param.type == "categorical":
                # check to see if all the provided options are inlcuded in the param_space options
                provided_options = self.data.iloc[:, i].unique().tolist()
                if not set(provided_options).issubset(param.options):
                    message = f"Set of options for categorcial value {param.name} does not match the set of options found in the data!"
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

    def to_disk(self, folder="custom_dataset"):
        """Save the dataset to disk in the format expected by Olympus for its own datasets. This can be useful if you
        plan to upload the dataset to the community datasets available online.

        Args:
            folder (str): Folder in which to save the dataset files.
        """

        # create the folder
        os.mkdir(folder)

        # save the numeric data
        self.data.to_csv(f"{folder}/data.csv", header=False, index=False)

        # save the description
        self._generate_description()
        with open(f"{folder}/description.txt", "w") as f:
            f.write(self._description)

        # save the config file
        _config = {}
        _config["constraints"] = {"parameters": "none", "measurements": "none"}

        _config["parameters"] = []
        for param in self.param_space.parameters:
            d = {
                "name": param.name,
                "type": param.kind,
                "lower": param.low,
                "upper": param.high,
            }
            _config["parameters"].append(d)

        _config["measurements"] = []
        for target in self._targets:
            t = {"name": target, "type": "continuous"}
            _config["measurements"].append(t)

        with open(f"{folder}/config.json", "w") as f:
            f.write(json.dumps(_config, indent=4, sort_keys=True))

    def _generate_description(self):
        _description = []
        if self.kind is None:
            _description.append("Custom Dataset\n")
        else:
            _description.append(f"{self.kind}\n")

        _description.append("=========================================")
        _description.append("                Summary")
        _description.append("-----------------------------------------")
        _description.append(f"    Number of Samples       {self.size:>10}")
        _description.append(
            f"    Dimensionality          {len(self.feature_names):>10}"
        )
        _description.append(f"    Features:")
        for param in self.param_space.parameters:
            _description.append(
                f"        {param.name:<10}          {param.kind:>10}"
            )
        _description.append(f"    Targets:")
        for target in self._targets:
            _description.append(f"        {target:<10}          continuous")

        self._description = "\n".join(_description)

    def run(self, params, return_paramvector=False):
        """run method to allow lookup of target values for fully categorical
        parameter spaces. This method is named run to make it interchangable with
        the emulator and surface objects within Olympus, such that it can be used in the
        higher level Evaluator class for optimization runs and larger benchmarks

        Args:
            params (ndarray): 2d array which contains the input parameters
            return_paramvector (bool): return an Olympus ParameterVector object
                or a list of lists. Default is False

        Returns:
            values (ParamVector): output value referenced from the lookup table. Returns
                a list of num samples elements.
        """
        if self.dataset_type is not "full_cat":
            message = f"Value lookup only supported for fully categorical/discrete parameter spaces"
            Logger.log(message, "FATAL")

        # check the type of params that have been passed, convert to list of
        # arrays to be processed in the lookup step
        if isinstance(params, np.ndarray):
            if len(params.shape) == 2:
                params = list(params)  # multiple observations
            elif len(params.shape) == 1:
                params = [params]  # assuming a single observation
            else:
                message = f"You can pass either a 1d or 2d np.ndarray for argument params. You have passed a {len(params.shape)}d np.ndarray."
                Logger.log(message, "ERROR")

        elif isinstance(params, list):
            if type(params[0]) in [str, int, float]:
                # assume we have a single value passed
                params = [params]
            elif type(params[0]) in [list, np.ndarray]:
                # assume multiple parameters are passed already in array form
                pass
            elif type(params[0]) == ParameterVector:
                # list of ParamVectors, convert to list of arrays
                params = [param.to_array() for param in params]

        elif isinstance(params, ParameterVector):
            # assuming single ParameterVector object, convert to array
            params = [params.to_array()]
        else:
            Logger.log(
                "Params type not understood. Accepted types are: np.ndarray, list, and ParameterVector",
                "FATAL",
            )

        # assert that have the correct number of parameters for each sample
        assert np.all(
            [len(param) == len(self.feature_names) for param in params]
        )

        values = []
        for param in params:
            sub_df = self.data.copy()
            for name, space, val in zip(
                self.feature_names, self.param_space, param
            ):
                if space.type in ["continuous", "discrete"]:
                    val = round(
                        val, 5
                    )  # five should be max precision, but this may cause errors
                sub_df = sub_df.loc[(sub_df[name] == val), :]
            if not sub_df.shape[0] == 1:
                message = f"Could not find lookup value for parameter setting {param}"
                Logger.log(message, "FATAL")

            value_objs = []
            # iterate over all objectives/targets
            for target_name in self.target_names:
                val = sub_df[target_name].tolist()[0]
                # if return_paramvector:
                #     value_obj = ParameterVector().from_dict({target_name: val})
                # else:
                #     value_obj = val
                value_objs.append(val)
            if return_paramvector:
                value_objs = ParameterVector().from_dict(
                    {
                        target_name: val
                        for target_name, val in zip(
                            self.target_names, value_objs
                        )
                    }
                )
            values.append(value_objs)

        return values

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
    def features_dim_ohe(self):
        dim = 0
        for param in self.param_space:
            if param.type in ["continuous", "discrete"]:
                dim += 1
            elif param.type == "categorical":
                dim += len(param.options)
        return dim

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
    def _get_descriptors(self, param):
        """if categorical parameter options have descriptors have desc,
        list of lists containing descriptors
        """
        if not param["descriptors"]:
            # no descriptors, return None for each option
            desc = [None for _ in param["options"]]
        else:
            # we have some descritptors
            desc = []
            assert not type(self.descriptors) == type(None)
            param_desc = self.descriptors[
                self.descriptors["param"] == param["name"]
            ]
            for option in param["options"]:
                d = param_desc[param_desc["option"] == option][
                    "value"
                ].tolist()
                desc.append(d)
        return desc

    def _create_param_space(self, config):
        self.param_space = ParameterSpace()
        for param in config["parameters"]:
            if param["type"] == "categorical":
                desc_ = self._get_descriptors(param)
                self.param_space.add(
                    ParameterCategorical(
                        name=param["name"],
                        options=param["options"],
                        descriptors=desc_,
                    )
                )
            # continuous or categorical
            else:
                self.param_space.add([Parameter().from_dict(param)])

    def _create_aux_param_space(self, param_dim):
        self.aux_param_space = ParameterSpace()
        for i in range(param_dim - 1):
            self.aux_param_space.add(
                ParameterContinuous(name=f"param_{i}", low=0.0, high=1.0)
            )

    def _create_value_space(self, config):
        self.value_space = ParameterSpace()
        self.value_space.add(
            [
                Parameter().from_dict(feature)
                for feature in config["measurements"]
            ]
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

    _validate_dataset_args(
        kind=kind, data=None, columns=None, target_names=None
    )
    datasets_path = os.path.dirname(os.path.abspath(__file__))

    # load description
    with open(
        "".join(f"{datasets_path}/dataset_{kind}/description.txt")
    ) as txtfile:
        description = txtfile.read()

    # load info on features/targets
    with open(
        "".join(f"{datasets_path}/dataset_{kind}/config.json"), "r"
    ) as content:
        config = json.loads(content.read())

    # load data
    csv_file = "".join(f"{datasets_path}/dataset_{kind}/data.csv")
    try:
        data = read_csv(csv_file, header=None).to_numpy()
    except FileNotFoundError:
        Logger.log(f"Could not find data.csv for dataset {kind}", "FATAL")

    # load descriptors
    csv_file = "".join(f"{datasets_path}/dataset_{kind}/descriptors.csv")
    # try:
    #     descriptors = read_csv(csv_file, header=None).to_numpy()
    # except FileNotFoundError:
    #     Logger.log(f'No descriptors found for dataset {kind}', 'WARNING')
    #     descriptors = None
    if os.path.isfile(csv_file):
        descriptors = read_csv(csv_file, header=None).to_numpy()
    else:
        descriptors = None

    return data, config, description, descriptors


def _validate_dataset_args(kind, data, columns, target_names):
    if kind is not None:
        # -----------------------------------
        # check that a correct name is passed
        # -----------------------------------
        # TODO: reduce redundant code by importing the list from where we have it already
        module_path = os.path.dirname(os.path.abspath(__file__))
        olympus_datasets = []
        for dir_name in glob(f"{module_path}/dataset_*"):

            if "/" in dir_name:
                dir_name = dir_name.split("/")[-1][8:]
            elif "\\" in dir_name:
                dir_name = dir_name.split("\\")[-1][8:]

            olympus_datasets.append(dir_name)
        if kind not in olympus_datasets:
            message = (
                "Could not find dataset `{0}`. Please choose from one of the available "
                "datasets: {1}.".format(
                    kind, ", ".join(list(olympus_datasets))
                )
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
