#!/usr/bin/env python

import os
import pickle
import shutil
from copy import deepcopy
from glob import glob
from tempfile import TemporaryDirectory

import numpy as np

try:
    from sklearn.metrics import r2_score, accuracy_score
except ModuleNotFoundError:
    # import minimal version
    from olympus.utils.misc import r2_score

from olympus import Logger, __emulator_path__, __scratch__, __version__
from olympus.datasets.dataset import Dataset, _validate_dataset_args
from olympus.models.abstract_model import AbstractModel
from olympus.models.model import Model, _validate_model_kind
from olympus.objects import Object, ParameterVector
from olympus.utils.data_transformer import DataTransformer, cat_param_to_feat
from olympus.utils.misc import r2_score


# =========================
# Main Class of This Module
# =========================
class Emulator(Object):
    """generic experiment emulator

    This class is intended to provide the interface to the user.

    Random notes:
    - emulators are uniquely determined via dataset + model + emulator_id
    """

    def __init__(
        self,
        dataset=None,
        model=None,
        feature_transform="identity",
        target_transform="identity",
    ):
        """Experiment emulator.

        Args:
            dataset (str, Dataset): dataset used to train a model. Either a string, in which case a standard dataset
                is loaded, or a Dataset object. To see the list of available datasets ...
            model (str, Model): the model used to create the emulator. Either a string, in which case a default model
                is loaded, or a Model object. To see the list of available models ...
            feature_transform (str, list): the data transform to be applied to the features. See DataTransformer for the
                available transformations.
            target_transform (str, list): the data transform to be applied to the targets. See DataTransformer for the
                available transformations.
        """

        # ------------------------------------------------------------
        # if dataset and model are strings ==> load emulator from file
        # ------------------------------------------------------------
        if type(dataset) == str and type(model) == str:
            # check dataset string
            _validate_dataset_args(
                kind=dataset, data=None, columns=None, target_names=None
            )
            # check model string
            _validate_model_kind(model)
            Logger.log(
                f"Loading emulator using a {model} model for the dataset {dataset}...",
                "INFO",
            )
            self._load(f"{__emulator_path__}/emulator_{dataset}_{model}")

            self.parameter_constriants = self.dataset.constraints["parameters"]
            self.aux_param_space = self.dataset.aux_param_space

        # -----------------------------------------
        # otherwise, assume it is a custom emulator
        # -----------------------------------------
        else:
            Object.__init__(**locals())

            if dataset is not None:
                self._set_dataset(dataset)
            if model is not None:
                self._set_model(model)

            # other attributes we will use
            self._version = __version__
            self._ghost_model = deepcopy(self.model)
            self.is_trained = False
            self.cross_val_performed = False
            self.cv_scores = None
            self.model_scores = None
            self.emulator_to_save = None


            self.feature_transformer = DataTransformer(
                transformations=self.feature_transform
            )
            # if we have ordinal targets, force the traget transform to be 'identity'
            # if self.task=='ordinal' and target_transform!='identity' :
            #     message = 'Setting "target_transform" to "identity" for ordinal targets'
            #     Logger.log(message, 'WARNING')
            #     self.target_transform = 'identity'
            self.target_transformer = DataTransformer(
                transformations=self.target_transform
            )

        # create tmp dir to store model files
        # also if we are loading a model (the user could call 'train' again)
        self._scratch_dir = TemporaryDirectory(
            dir=f"{__scratch__}", prefix="emulator_"
        )

    def __str__(self):
        if self.dataset is not None and self.model is not None:
            return f"<Emulator ({self.dataset}, model={self.model})>"
        elif self.dataset is not None:
            return f"<Emulator ({self.dataset})>"
        elif self.model is not None:
            return f"<Emulator (model={self.model})>"
        else:
            return f"<Emulator (unspecified)>"

    @property
    def task(self):
        return self.dataset.task

    @property
    def metric_names(self):
        return self.dataset.metric_names


    @property
    def goal(self):
        return self.dataset.goal

    @property
    def param_space(self):
        return self.dataset.param_space

    @property
    def value_space(self):
        return self.dataset.value_space

    @property 
    def task(self):
        return self.dataset.task

    # ===========
    # Set Methods
    # ===========
    def _set_dataset(self, dataset):
        """registers a dataset for emulator

        Args:
            dataset (str): name of available dataset, or Dataset object.
        """
        if type(dataset) == str:
            self.dataset = Dataset(kind=dataset)
        elif isinstance(dataset, Dataset):
            self.dataset = dataset
        else:
            raise NotImplementedError

        # check that the param_space is defined
        if self.dataset.param_space is None:
            message = (
                "The param_space information is not present in the Dataset object provided. Please use "
                "Dataset.set_param_space to define the type of variables present in the dataset before "
                "instantiating the Emulator."
            )
            Logger.log(message, "ERROR")

    def _set_model(self, model):
        """registers a model for emulator

        Args:
            model (str): name of available model, or a model object
        """
        if type(model) == str:
            self.model = Model(kind=model)
        elif isinstance(model, AbstractModel):
            self.model = model
        else:
            raise NotImplementedError
        # self.param_space is taken from self.dataset.param_space, so...
        if self.dataset is None:
            self.model.set_param_space(None)
            self.model.set_value_space(None)
        else:
            self.model.set_param_space(self.param_space)
            self.model.set_value_space(self.value_space)



    def transform_cat_params(self, params):
        """transformed parameter space representation of categorical variables
        to one-hot-encoded representation for emulator
        """

        transformed_features = []
        for param in params:
            transformed_feature = []
            for p, space in zip(param, self.param_space):
                if space.type == "categorical":
                    element = cat_param_to_feat(space, p)
                    transformed_feature.extend(element)
                else:
                    element = p
                    transformed_feature.append(element)
            transformed_features.append(transformed_feature)

        return np.array(transformed_features)


    def transform_ordinal_targets(self, targets):
        # NOTE: this will only support single target ordinal parameters
        options = self.value_space[0].options
        transformed_targets = np.zeros( (targets.shape[0], len(options)) )
        int_list = [options.index(t[0]) for t in targets]

        for ix, target in enumerate(int_list):
            transformed_targets[ix, 0:target+1] = 1

        return transformed_targets


    # =========================
    # Train and Predict Methods
    # =========================
    def cross_validate(self, rerun=False, plot=False):
        # TODO: allow setting verbosity: verbose=True/False will be enough
        """Performs cross validation on the emulator dataset, using the emulator model. The number of folds used is
        defined in the Dataset object.

        Args:
            rerun (bool): whether to run cross validation again, in case it had already been performed.

        Returns:
            scores (dict): dictionary with the list of train and validation R2 scores.

        """
        """Perform cross validation.

        Returns (dict): dictionary containing the training and test R2 scores for all folds.

        """

        if self.cross_val_performed is True and rerun is False:
            message = (
                "Cross validation has already been performed for this Emulator. You can see its results in "
                "`self.cv_scores`. If you would like to rerun cross validation and overwrite the previous "
                "results, set `rerun` to True"
            )
            Logger.log(message, "FATAL")


        # if we have a regression problem, use r2 and rmsd
        # if we have an ordinal problem, use accuracy and TODO: decide on the second metric

        training_metric1_scores = np.empty(self.dataset.num_folds)
        valid_metric1_scores = np.empty(self.dataset.num_folds)
        training_metric2_scores = np.empty(self.dataset.num_folds)
        valid_metric2_scores = np.empty(self.dataset.num_folds)



        # get scaled train/valid sets
        # NOTE: we do not want to use the self.transformers, because for 'run' we want to use the transformers
        # trained in 'train'. If we reset the Transformers here, then if a user calls 'cross_validate' after 'train'
        # we end up using the wrong transformers in 'run'
        feature_transformer = DataTransformer(
            transformations=self.feature_transform
        )
        # if we have an ordinal problem, make sure the
        target_transformer = DataTransformer(
            transformations=self.target_transform
        )


        # ---------------------------------------
        # Iterate over the cross validation folds
        # ---------------------------------------
        for fold in range(self.dataset.num_folds):
            # get the train/valid sets
            # NOTE: we keep the features as Dataset objects, as these are needed for possible periodic transformations
            # TODO: expend the above also to targets? Right now param_space does not describe what type of variable
            #  the targets are

            train_features = Dataset(
                data=self.dataset.cross_val_sets_features[fold][0]
            )
            train_features.set_param_space(self.dataset.param_space)
            train_features = train_features.data.to_numpy()
            valid_features = self.dataset.cross_val_sets_features[fold][
                1
            ].to_numpy()

            if self.task == 'regression':
                train_targets = (
                    self.dataset.cross_val_sets_targets[fold][0]
                    .to_numpy()
                    .astype(float)
                )
                valid_targets = (
                    self.dataset.cross_val_sets_targets[fold][1]
                    .to_numpy()
                    .astype(float)
                )
            elif self.task == 'ordinal':
                # convert the ordinal targets to the proper form
                train_targets = self.transform_ordinal_targets( self.dataset.cross_val_sets_targets[fold][0].values )
                valid_targets = self.transform_ordinal_targets( self.dataset.cross_val_sets_targets[fold][1].values )


            # transform categorical variables to one-hot-encoded vectors, if any exist
            train_features = self.transform_cat_params(train_features)
            valid_features = self.transform_cat_params(valid_features)

            feature_transformer.train(train_features)
            target_transformer.train(train_targets)

            train_features_scaled = feature_transformer.transform(
                train_features
            )
            valid_features_scaled = feature_transformer.transform(
                valid_features
            )
            train_targets_scaled = target_transformer.transform(train_targets)
            valid_targets_scaled = target_transformer.transform(valid_targets)


            # define scope and make a copy of the model for the cross validation
            model_fold = deepcopy(
                self._ghost_model
            )  # the model we will use for training the fold
            model_fold.scope = f"Fold_{fold}"
            model_path = f"{self._scratch_dir.name}/{model_fold.scope}"
            # TODO/QUESTION: in case we are overwriting the output of a previous call, should we first remove the folder
            #  to make sure to have a cleared path?
            if not os.path.exists(model_path):
                os.makedirs(model_path)


            Logger.log(f">>> Training model on fold #{fold}...", "INFO")
            # this call to train is the neural network model method, not the emulator
            (
                mdl_train_metric1,
                mdl_valid_metric1,
                mdl_train_metric2,
                mdl_test_metric2,
            ) = model_fold.train(
                train_features=train_features_scaled,
                train_targets=train_targets_scaled,
                valid_features=valid_features_scaled,
                valid_targets=valid_targets_scaled,
                model_path=model_path,
                plot=plot,
            )

            # store performance of fold
            training_metric1_scores[fold] = mdl_train_metric1
            valid_metric1_scores[fold] = mdl_valid_metric1
            training_metric2_scores[fold] = mdl_train_metric2
            valid_metric2_scores[fold] = mdl_test_metric2
            # write file to indicate training is complete and add R2 in there
            with open(f"{model_path}/training_completed.info", "w") as content:
                content.write(
                    f"Train {self.metric_names[0].upper()}={mdl_train_metric1}\nValidation {self.metric_names[0].upper()}={mdl_valid_metric1}\n"
                    f"Train {self.metric_names[1].upper()}={mdl_train_metric2}\nValidation {self.metric_names[1].upper()}={mdl_test_metric2}\n"
                )


            # print some info to screen
            Logger.log(
                f"Performance statistics based on transformed data "
                f"[{self.feature_transform}, {self.target_transform}]:",
                "INFO",
            )
            cv_metric1_score_mean = np.mean(valid_metric1_scores)
            cv_metric1_score_stderr = np.std(valid_metric1_scores) / np.sqrt(
                (len(valid_metric1_scores) - 1)
            )
            cv_metric2_score_mean = np.mean(valid_metric2_scores)
            cv_metric2_score_stderr = np.std(valid_metric2_scores) / np.sqrt(
                (len(valid_metric2_scores) - 1)
            )


        Logger.log(
            "Validation {}: {0:.4f} +/- {1:.4f}".format(
                self.metrics_names[0].upper(), cv_metric1_score_mean, cv_metric1_score_stderr
            ),
            "INFO",
        )
        Logger.log(
            "Validation {}: {0:.4f} +/- {1:.4f}".format(
                self.metrics_names[1].upper(), cv_rmsd_score_mean, cv_rmsd_score_stderr
            ),
            "INFO",
        )

        self.cross_val_performed = True
        self.cv_scores = {
            f"train_{self.metric_names[0]}": training_metric1_scores,
            f"validate_{self.metric_names[0]}": valid_metric1_scores,
            f"train_{self.metric_names[1]}": training_metric2_scores,
            f"validate_{self.metric_names[1]}": valid_metric2_scores,
        }

        return self.cv_scores

    def train(self, plot=False, retrain=False):
        # TODO: allow setting verbosity: verbose=True/False will be enough
        """Trains the model on the emulator dataset, using the emulator model. The train/test split is defined in the
        Dataset object `emulator.dataset`. Note that the test set is used for testing the model performance, and for
        early stopping.

        Args:
            plot (bool):
            retrain (bool): whether to retrain the model, in case it had already been trained.

        Returns:
            scores (dict): dictionary with the train and test R2 scores.

        """
        # check if this emulator has already been trained
        if self.is_trained is True and retrain is False:
            message = (
                "The Emulator is already trained. If you would like to overwrite the already trained emulator, "
                "set `retrain` to True"
            )
            Logger.log(message, "FATAL")

        # get the train/test sets
        # NOTE: we keep the features as Dataset objects, as these are needed for possible periodic transformations
        # TODO: expend the above also to targets? Right now param_space does not describe what type of variable
        #  the targets are
        train_features = Dataset(data=self.dataset.train_set_features)
        train_features.set_param_space(self.dataset.param_space)
        train_features = train_features.data.to_numpy()
        test_features = self.dataset.test_set_features.to_numpy()

        if self.task == 'regression':
            # continuous-valued targets
            train_targets = self.dataset.train_set_targets.to_numpy().astype(float)
            test_targets = self.dataset.test_set_targets.to_numpy().astype(float)

        elif self.task == 'ordinal':
            # ordinal-valued targets
            # convert the ordinal targets to the proper form
            train_targets = self.transform_ordinal_targets( self.dataset.train_set_targets.to_numpy() )
            test_targets = self.transform_ordinal_targets( self.dataset.test_set_targets.to_numpy() )


        # transform categorical variables to one-hot-encoded vectors, if any exist
        train_features = self.transform_cat_params(train_features)
        test_features = self.transform_cat_params(test_features)

        # get scaled train/valid sets. These are also the DataTransformer objects we keep as they are needed in 'run'
        self.feature_transformer.train(train_features)
        self.target_transformer.train(train_targets)

        train_features_scaled = self.feature_transformer.transform(
            train_features
        )
        test_features_scaled = self.feature_transformer.transform(
            test_features
        )
        train_targets_scaled = self.target_transformer.transform(train_targets)
        test_targets_scaled = self.target_transformer.transform(test_targets)

        # define path where to store model
        model_path = f"{self._scratch_dir.name}/Model"
        # TODO/QUESTION: in case we are overwriting the output of a previous call, should we first remove the folder
        #  to make sure to have a cleared path?
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        # Train
        Logger.log(
            ">>> Training model on {0:.0%} of the dataset, testing on {1:.0%}...".format(
                (1 - self.dataset.test_frac), self.dataset.test_frac
            ),
            "INFO",
        )
        (
            mdl_train_metric1,
            mdl_test_metric1,
            mdl_train_metric2,
            mdl_test_metric2,
        ) = self.model.train(
            train_features=train_features_scaled,
            train_targets=train_targets_scaled,
            valid_features=test_features_scaled,
            valid_targets=test_targets_scaled,
            model_path=model_path,
            plot=plot,
        )

        # write file to indicate training is complete and add R2 in there
        with open(f"{model_path}/training_completed.info", "w") as content:
            content.write(
                f"Train {self.metric_names[0].upper()}={mdl_train_metric1}\nValidation {self.metric_names[0].upper()}={mdl_test_metric1}\n"
                f"Train {self.metric_names[1].upper()}={mdl_train_metric2}\nValidation {self.metric_names[1].upper()}={mdl_test_metric2}\n"
            )

        Logger.log(
            f"Performance statistics based on transformed data "
            f"[{self.feature_transform}, {self.target_transform}]:",
            "INFO",
        )
        Logger.log(f"Train {self.metric_names[0].upper()} "+"  Score: {0:.4f}".format(mdl_train_metric1), "INFO")
        Logger.log(f"Test  {self.metric_names[0].upper()} "+"  Score: {0:.4f}".format(mdl_test_metric1), "INFO")
        Logger.log(f"Train {self.metric_names[1].upper()} "+"  Score: {0:.4f}".format(mdl_train_metric2), "INFO")
        Logger.log(f"Test  {self.metric_names[1].upper()} "+"  Score: {0:.4f}\n".format(mdl_test_metric2), "INFO")

        # set is_trained to True
        self.is_trained = True

        # show stats on untransformed samples
        # -----------------------------------
        #y_train = self.dataset.train_set_targets.to_numpy()
        y_train_pred = self.run(
           features=self.dataset.train_set_features.to_numpy(), num_samples=10, return_ordinal_label=False
        )

        #y_test = self.dataset.test_set_targets.to_numpy()
        y_test_pred = self.run(
           features=self.dataset.test_set_features.to_numpy(), num_samples=10, return_ordinal_label=False
        )

        if self.task == 'regression':   
            train_metric1 = r2_score(train_targets, y_train_pred)
            train_metric2 = np.sqrt(
                np.mean((train_targets.flatten() - y_train_pred.flatten()) ** 2)
            )
            test_metric1 = r2_score(test_targets, y_test_pred)
            test_metric2 = np.sqrt(
                np.mean((test_targets.flatten() - y_test_pred.flatten()) ** 2)
            )
 
        elif self.task == 'ordinal':
            # TODO: implement these stats properly
            train_metric1 = self.compute_acc_ordinal(train_targets, y_train_pred)
            train_metric2 = np.sqrt(
                np.mean((train_targets.flatten() - y_train_pred.flatten()) ** 2)
            )
            test_metric1 = self.compute_acc_ordinal(test_targets, y_test_pred)
            test_metric2 = np.sqrt(
                            np.mean((test_targets.flatten() - y_test_pred.flatten()) ** 2)
                        )


        Logger.log(f"Performance statistics based on original data:", "INFO")
        Logger.log(f"Train {self.metric_names[0].upper()} "+"  Score: {0:.4f}".format(train_metric1), "INFO")
        Logger.log(f"Test  {self.metric_names[0].upper()} "+"  Score: {0:.4f}".format(test_metric1), "INFO")
        Logger.log(f"Train {self.metric_names[1].upper()} "+"  Score: {0:.4f}".format(train_metric2), "INFO")
        Logger.log(f"Test  {self.metric_names[1].upper()} "+"  Score: {0:.4f}\n".format(test_metric2), "INFO")

        # save and return scores

        self.model_scores = {
            f"train_{self.metric_names[0]}": train_metric1,
            f"test_{self.metric_names[0]}": test_metric1,
            f"train_{self.metric_names[1]}": train_metric2,
            f"test_{self.metric_names[1]}": test_metric2,
        }
        

        return self.model_scores

    @staticmethod
    def compute_acc_ordinal(y_true, y_pred):
        true_labels = (y_true>0.5).cumprod(axis=1).sum(axis=1)-1.
        pred_labels = (y_pred>0.5).cumprod(axis=1).sum(axis=1)-1.

        return accuracy_score(true_labels, pred_labels)


    def run(self, features, num_samples=1, return_paramvector=False, return_ordinal_label=True):
        """Run the emulator and return a value given the features provided.

        Args:
            features (ndarray): 2d array with the input features used for the predictions
            num_samples (int): number of samples to average. only useful for probabilistic models
            return_paramvector (bool): Whether to return a ``ParameterVector`` object instead of a list of lists.
                Default is False.

        Returns:
            y (float or array): model prediction(s).
        """

        # first check if we have a trained model
        if not self.is_trained:
            message = "This emulator has not been trained yet. Please train the emulator before you can use it for prediction."
            Logger.log(message, "ERROR")

        # check the type of features given and convert to a list of lists
        # (# samples, # features) to be processed

        if isinstance(features, np.ndarray):
            if len(features.shape) == 2:
                features = [
                    list(feature) for feature in features
                ]  # multiple observations, convert to list
            elif len(features.shape) == 1:
                features = [list(features)]  # assume single sample
            else:
                message = f"You can pass either a 1d or 2d np.ndarray for argument features. You have passed a {len(features.shape)}d np.ndarray."
                Logger.log(message, "ERROR")

        elif isinstance(features, list):
            if type(features[0]) in [str, int, float]:
                # assume we have a single value passed
                features = [features]
            elif type(features[0]) in [list, np.ndarray]:
                # assume multiple parameters are passed already in array form
                features = [list(feature) for feature in features]
            elif type(features[0]) == ParameterVector:
                # list of ParamVectors, convert to list of arrays
                features = [feature.to_list() for feature in features]

        elif isinstance(features, ParameterVector):
            # assuming single ParameterVector object, convert to lists
            features = [features.to_list()]

        else:
            Logger.log(
                "Features type not understood. Accepted types are: np.ndarray, list, and ParameterVector",
                "FATAL",
            )

        # validate the provided features
        if not np.all(
            [len(feature) == len(self.param_space)] for feature in features
        ):
            message = (
                "Dimensions of provided features (%d) did not match expected dimension (%d)"
                % (features.shape[1], len(self.param_space))
            )
            Logger.log(message, "ERROR")

        for feature in features:
            if not self.param_space.validate(feature):
                message = "Not all parameters are within bounds"
                Logger.log(message, "WARNING")

        # convert categorical params to ohe vectors
        features = self.transform_cat_params(features)

        # scale the features using the DataTransformer that was fit when training
        features_scaled = self.feature_transformer.transform(features)

        # predict, drawing a certain amount of samples
        y_pred_scaled = self.model.predict(
            features_scaled, num_samples=num_samples
        )

        # return the prediction after inverting the transform
        y_preds = self.target_transformer.back_transform(
            y_pred_scaled
        )  # this is a 2d array

        if self.task=='ordinal' and return_ordinal_label:
            # if ordinal parameter, convert it back to its predicted label 
            # NOTE: need to inflate dims here, this is kind of a hack
            y_preds = [ (y_preds>0.5).cumprod(axis=1).sum(axis=1)-1 ]

    

        # if we are not asking for a ParamVector, we can just return y_preds
        if return_paramvector is False:
            return y_preds

        # NOTE: while we do not allow batches or multiple objectives yet, this code is supposed to be able to support
        #  those
        y_pred_objects = (
            []
        )  # list of ParamVectors with all samples and objectives
        # iterate over all samples (if we returned a batch of predictions)
        for y_pred in y_preds:
            y_pred_object = ParameterVector()
            # iterate over all objectives/targets
            for ix, (target_name, y) in enumerate(zip(self.dataset.target_names, y_pred)):
                if self.task == 'ordinal':
                    # return the ordinal variable option in the string representation
                    str_option = self.dataset.value_space[ix]['options'][y] # index of the predicted label
                    y_pred_object.from_dict({target_name:str_option})
                else:
                    y_pred_object.from_dict({target_name: y})
            # append object to list
            y_pred_objects.append(y_pred_object)

        return y_pred_objects

    def save(self, path="./olympus_emulator", include_cv=False):
        """Save the emulator in a specified location. This will save the emulator object as a pickle file, and the
        associated TensorFlow model, in the specified location. The saved emulator can then be loaded with the
        `olympus.emulators.load_emulator` function.

        Args:
            path (str): relative path where to save the emulator.
            include_cv (bool): whether to include the cross validation models. Default is False.

        """

        # TODO: check if path corresponds to one of our emulators and in that case raise error?

        # create path, or overwrite existing one
        if os.path.exists(path):
            message = f"Overwriting existing emulator in {path}!"
            Logger.log(message, "WARNING")
            shutil.rmtree(path)
        os.makedirs(path)

        self.emulator_to_save = Emulator()
        for key in self.__dict__:
            #            skip = [
            #                'me',
            #                'indent',
            #                'props',
            #                'attrs',
            #                'max_prop_len',
            #                'dataset',
            #                'model',
            #                'feature_transform',
            #                'target_transform',
            #                '_version',
            #                '_ghost_model',
            #                'is_trained',
            #                'cross_val_performed',
            #                'cv_scores',
            #                'model_scores',
            #                'emulator_to_save',
            #                'feature_transformer',
            #                'target_transformer',
            #                '_scratch_dir']
            #            ]
            #            if key in skip:
            #                continue

            if key == "model":
                setattr(self.emulator_to_save, key, self._ghost_model)
            elif key != "emulator_to_save":
                setattr(self.emulator_to_save, key, self.__dict__[key])
        self.emulator_to_save.reset()

        # pickle emulator object
        with open(f"{path}/emulator.pickle", "wb") as f:
            pickle.dump(self.emulator_to_save, f)

        # copy over model files
        if self.is_trained is True:
            shutil.copytree(f"{self._scratch_dir.name}/Model", f"{path}/Model")
        else:
            message = "The emulator you are saving has not been trained. Its model-related files will not be written to disk."
            Logger.log(message, "WARNING")

        # if include_cv is True ==> copy also CV models
        if include_cv is True:
            if self.cross_val_performed is True:
                for folder in glob(f"{self._scratch_dir.name}/Fold*"):
                    dirname = folder.split("/")[-1]
                    shutil.copytree(folder, f"{path}/{dirname}")
            else:
                message = (
                    "You are trying to save the cross validation models for an emulator that did not perform "
                    "cross validation. No cross validation models to be saved."
                )
                Logger.log(message, "WARNING")

        # TODO: also dump some info in human readable format? json like we had before?

    def _load(self, emulator_folder):
        emulator = load_emulator(emulator_folder)
        self.__dict__.update(emulator.__dict__)


# ===============
# Other Functions
# ===============
def load_emulator(emulator_folder):
    """Loads a previously saved emulator.

    Args:
        emulator_folder (str): path to the folder where the emulator was saved.

    Returns:
        emulator (Emulator): emulator object loaded from file.

    """

    # check path exists
    if os.path.exists(emulator_folder) is False:
        message = f'Folder "{emulator_folder}" does not exist'
        Logger.log(message, "FATAL")

    try:
        with open(f"{emulator_folder}/emulator.pickle", "rb") as f:
            emulator_stored = pickle.load(f)
    except TypeError:
        Logger.log("Failed to restore emulator", "FATAL")

    emulator_to_load = Emulator()
    for key in emulator_stored.__dict__:
        if key in emulator_to_load.__dict__:
            emulator_to_load.add(key, emulator_stored.get(key))
        else:
            message = (
                f'Key "{key}" from Emulator of olympus v.{emulator_stored._version} not found in Emulator '
                f'class of olympus v.{__version__}). Attribute "{key}" will not be set.'
            )
            Logger.log(message, "WARNING")

    if emulator_to_load.is_trained is True:
        emulator_to_load.model._set_dims(
            features_dim=emulator_to_load.dataset.features_dim_ohe,
            targets_dim=emulator_to_load.dataset.targets_dim,
        )
        restored = emulator_to_load.model.restore(f"{emulator_folder}/Model")
        if restored is False:
            message = "failed to restore model"
            Logger.log(message, "ERROR")

    # NOTE: we are not leading any cross validation models for the moment. If CV was performed though, we still have
    # the data about the CV performance
    return emulator_to_load
