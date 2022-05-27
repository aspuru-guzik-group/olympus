#!/usr/bin/env python

import os
import time

import numpy as np
import silence_tensorflow

silence_tensorflow.silence_tensorflow()
import tensorflow as tf
from sklearn.metrics import mean_squared_error, r2_score

from olympus import Logger
from olympus.models import AbstractModel

# ===============================================================================

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# ===============================================================================


class WrapperTensorflowModel(AbstractModel):
    def __init__(self, *args, **kwargs):
        AbstractModel.__init__(self, *args, **kwargs)
        self.graph = None
        self.is_graph_constructed = False
        tf.compat.v1.reset_default_graph()

    def act_funcs(self, act_func_name):
        activation_functions = {
            "linear": lambda y: y,
            "leaky_relu": lambda y: tf.nn.leaky_relu(y, 0.2),
            "relu": lambda y: tf.nn.relu(y),
            "softmax": lambda y: tf.nn.softmax(y),
            "softplus": lambda y: tf.nn.softplus(y),
            "softsign": lambda y: tf.nn.softsign(y),
            "sigmoid": lambda y: tf.nn.sigmoid(y),
        }
        return activation_functions[act_func_name]

    def _generator(self, features, targets):
        batch_indices = np.random.randint(
            low=0, high=features.shape[0], size=self.batch_size
        )
        features_batch = features[batch_indices]
        targets_batch = targets[batch_indices]
        return features_batch, targets_batch

    def _set_dims(self, features_dim, targets_dim):
        self.features_dim = features_dim
        self.targets_dim = targets_dim

    # def _project_features(self, features, transformer):
    #    projections = []
    #    transformed_features = transformer.back_transform(features)
    #    for _, param_definition in enumerate(self.param_space):
    #        feature_values = features[:, _]
    #        if param_definition.is_periodic is True:
    #            # compute sine and cosine
    #            cosine = np.cos( 2 * np.pi * (transformed_features[:, _] - param_definition.low) / (param_definition.high - param_definition.low))
    #            sine   = np.sin( 2 * np.pi * (transformed_features[:, _] - param_definition.low) / (param_definition.high - param_definition.low))
    #            projections.append(cosine)
    #            projections.append(sine)
    #        elif param_definition.is_periodic is False:
    #            projections.append(feature_values)
    #   #             projections.append(transformed_features[:, _])
    #        else:
    #            raise NotImplementedError
    #    projections = np.array(projections).T
    #    return projections

    # def train(self, train_features, train_targets, valid_features, valid_targets, model_path,
    #          plot=False, feature_transformer=None, target_transformer=None):
    def train(
        self,
        train_features,
        train_targets,
        valid_features,
        valid_targets,
        model_path,
        plot=False,
    ):
        """

        Args:
            train_features: features of the training set.
            train_targets: targets of the training set.
            valid_features: features of the validation/test set, used for early stopping.
            valid_targets: targets of the validation/test set, used for early stopping.
            model_path (str): where the TensorFlow checkpoints will be written.
            plot (bool): whether to show scatter plots of the training progression.
            feature_transformer:
            target_transformer:

        Returns:
            max_train_r2: best r2 score on training set
            max_valid_r2: best r2 score on validation set
        """
        # train_features = self._project_features(train_features, feature_transformer)
        # valid_features = self._project_features(valid_features, feature_transformer)

        # compute needed stats for features and targets
        self._set_dims(
            features_dim=np.shape(train_features)[1],
            targets_dim=np.shape(train_targets)[1],
        )
        if not self.is_graph_constructed:
            self._build_inference()
        train_errors, valid_errors, losses = list(), list(), list()

        start_time = time.time()
        with self.graph.as_default():
            with self.sess.as_default():
                self.saver = tf.compat.v1.train.Saver()

                if plot:
                    import matplotlib.pyplot as plt

                    fig = plt.figure(figsize=(12, 6))
                    ax0 = plt.subplot2grid((1, 2), (0, 0))
                    ax1 = plt.subplot2grid((1, 2), (0, 1))
                    axs = [ax0, ax1]
                    plt.ion()

                if len(valid_targets) > self.batch_size:
                    valid_indices = np.random.randint(
                        low=0, high=len(valid_targets), size=self.batch_size
                    )
                else:
                    valid_indices = np.arange(len(valid_targets))

                if len(train_targets) > self.batch_size:
                    train_indices = np.random.randint(
                        low=0, high=len(train_targets), size=self.batch_size
                    )
                else:
                    train_indices = np.arange(len(train_targets))

                # print out info
                _print_header()
                for epoch in range(self.max_epochs):

                    (
                        train_features_batch,
                        train_targets_batch,
                    ) = self._generator(train_features, train_targets)
                    __, loss = self.sess.run(
                        [self.train_op, self.loss],
                        feed_dict={
                            self.tf_x: train_features_batch,
                            self.tf_y: train_targets_batch,
                        },
                    )
                    losses.append(loss)

                    if epoch % self.pred_int == 0:

                        # make a prediction on the validation set
                        valid_pred = self.predict(
                            features=valid_features[valid_indices],
                            num_samples=10,
                        )
                        valid_r2 = r2_score(
                            valid_targets[valid_indices], valid_pred
                        )
                        valid_rmsd = np.sqrt(
                            mean_squared_error(
                                valid_targets[valid_indices], valid_pred
                            )
                        )
                        valid_errors.append([valid_r2, valid_rmsd])

                        # make a prediction on the train set
                        train_pred = self.predict(
                            features=train_features[train_indices],
                            num_samples=10,
                        )
                        train_r2 = r2_score(
                            train_targets[train_indices], train_pred
                        )
                        train_rmsd = np.sqrt(
                            mean_squared_error(
                                train_targets[train_indices], train_pred
                            )
                        )
                        train_errors.append([train_r2, train_rmsd])

                        if plot:
                            for ax in axs:
                                ax.cla()

                            plot_train_targets = train_targets[train_indices]
                            plot_train_pred = train_pred
                            plot_valid_targets = valid_targets[valid_indices]
                            plot_valid_pred = valid_pred

                            plot_min_targets, plot_max_targets = (
                                np.amin(plot_train_targets),
                                np.amax(plot_train_targets),
                            )
                            ax0.plot(
                                [plot_min_targets, plot_max_targets],
                                [plot_min_targets, plot_max_targets],
                                color="#444444",
                            )
                            ax0.plot(
                                plot_train_targets,
                                plot_train_pred,
                                marker="o",
                                ls="",
                            )
                            ax0.plot(
                                plot_valid_targets,
                                plot_valid_pred,
                                marker="o",
                                ls="",
                            )
                            ax1.plot(np.array(train_errors)[:, 0])
                            ax1.plot(np.array(valid_errors)[:, 0])
                            plt.pause(0.05)

                        min_rmsd_index = np.argmin(
                            np.array(valid_errors)[:, 1]
                        )
                        if (
                            len(valid_errors) - min_rmsd_index
                            > self.es_patience
                        ):
                            break

                        newline = f"{epoch:>15}{train_r2:>15.3f}{train_rmsd:>15.3f}{valid_r2:>15.3f}{valid_rmsd:>15.3f}"
                        # the latest model is the best ==> save it and tag it on screen
                        if min_rmsd_index == len(valid_errors) - 1:
                            self.saver.save(
                                self.sess, f"{model_path}/model.ckpt"
                            )
                            newline += " *"
                        Logger.log(newline, "INFO")

                # report the train and valid performance of the model we saved
                # Note we saved the best model based on the lowest RMSE
                mdl_train_r2 = np.array(train_errors)[min_rmsd_index, 0]
                mdl_valid_r2 = np.array(valid_errors)[min_rmsd_index, 0]
                mdl_train_rmsd = np.array(train_errors)[min_rmsd_index, 1]
                mdl_valid_rmsd = np.array(valid_errors)[min_rmsd_index, 1]

        Logger.log(
            f"Training completed in {round(time.time() - start_time, 2)} seconds.",
            "INFO",
        )
        Logger.log("=" * 75 + "\n", "INFO")
        return mdl_train_r2, mdl_valid_r2, mdl_train_rmsd, mdl_valid_rmsd

    def restore(self, model_path):
        if not self.is_graph_constructed:
            tf.compat.v1.reset_default_graph()
            self._build_inference()
        with self.graph.as_default():
            with self.sess.as_default():
                try:
                    self.saver.restore(self.sess, model_path + "/model.ckpt")
                    return True
                # TODO: I get the error 'tf has no attribute python' here
                except tf.python.framework.errors_impl.InvalidArgumentError:
                    return False

    def predict(self, features, num_samples=1):
        # features = self._project_features(features, feature_transformer)
        # make sure the dimensionality of the input matches that used for training

        if features.shape[1] != self.features_dim:
            raise ValueError(
                "dimensionality of input features provided does not match that of the training dataset"
            )

        with self.sess.as_default():
            pred = np.empty((num_samples, len(features), self.targets_dim))
            resolution = divmod(len(features), self.batch_size)
            res = [self.batch_size for i in range(resolution[0])]
            res.append(resolution[1])
            res = list(filter((0).__ne__, res))
            res_cu = [i * self.batch_size for i in range(len(res))]
            res_stp = [res_cu[_] + re for _, re in enumerate(res)]

            for batch_iter, size in enumerate(res):
                start, stop = res_cu[batch_iter], res_stp[batch_iter]
                X_test_batch = None
                if size == self.batch_size:
                    X_test_batch = features[start:stop]
                elif size != self.batch_size:
                    X_test_batch = np.concatenate(
                        (
                            features[start:stop],
                            np.random.choice(
                                features[:, 0],
                                size=(
                                    self.batch_size - size,
                                    features.shape[1],
                                ),
                            ),
                        ),
                        axis=0,
                    )

                for _ in range(num_samples):
                    predic = self.sess.run(
                        self.y_pred, feed_dict={self.tf_x: X_test_batch}
                    )
                    pred[_, start:stop] = predic[:size]

            pred = np.mean(pred, axis=0)
            return pred  # return the scaled prediction (Emulator will back-transform)


def _print_header():
    Logger.log(
        "    =======================================================================",
        "INFO",
    )
    Logger.log(
        "{0:>15}{1:>15}{2:>15}{3:>15}{4:>15}".format(
            "Epoch", "Train R2", "Train RMSD", "Test R2", "Test RMSD"
        ),
        "INFO",
    )
    Logger.log(
        "    =======================================================================",
        "INFO",
    )
