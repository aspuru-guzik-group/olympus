#!/usr/bin/env python

import silence_tensorflow

silence_tensorflow.silence_tensorflow()
import tensorflow as tf
import tensorflow_probability as tfp

from olympus.models import WrapperTensorflowModel

tfd = tfp.distributions

# ===============================================================================


class BayesNeuralNet(WrapperTensorflowModel):

    ATT_KIND = {"type": "string", "default": "BayesNeuralNet"}

    def __init__(
        self,
        scope="model",
        hidden_depth=3,
        hidden_nodes=48,
        hidden_act="leaky_relu",
        out_act="linear",
        learning_rate=1e-3,
        pred_int=100,
        reg=0.001,
        es_patience=100,
        max_epochs=100000,
        batch_size=20,
    ):
        """Bayesian Neural Network model.

        Args:
            scope (str): TenforFlow scope.
            hidden_depth (int): Number of hidden layers.
            hidden_nodes (int): Number of hidden nodes per layer.
            hidden_act (str): Hidden activation function. Available options are 'linear', 'leaky_relu', 'relu',
                'softmax', 'softplus', 'softsign', 'sigmoid'.
            out_act (str): Output activation function. Available options are 'linear', 'leaky_relu', 'relu',
                'softmax', 'softplus', 'softsign', 'sigmoid'.
            learning_rate (float): Learning rate.
            pred_int (int): Frequency with which we make predictions on the validation/training set (in number of epochs).
            reg (float): ???
            es_patience (int): Early stopping patience.
            max_epochs (int): Maximum number of epochs allowed.
            batch_size (int): Size batches used for training.
        """

        WrapperTensorflowModel.__init__(**locals())

    def _build_inference(self):
        self.graph = tf.Graph()
        self.is_graph_constructed = True

        activation_hidden = self.act_funcs(self.hidden_act)
        activation_out = self.act_funcs(self.out_act)

        with self.graph.as_default():

            self.tf_x = tf.compat.v1.placeholder(
                tf.float32, [self.batch_size, self.features_dim]
            )
            self.tf_y = tf.compat.v1.placeholder(
                tf.float32, [self.batch_size, self.targets_dim]
            )

            with tf.name_scope(self.scope):
                self.layers = [
                    tfp.layers.DenseLocalReparameterization(
                        self.hidden_nodes, activation=activation_hidden
                    )
                    for _ in range(self.hidden_depth)
                ]
                self.layers.append(
                    tfp.layers.DenseLocalReparameterization(
                        self.targets_dim, activation=activation_out
                    )
                )

                self.neural_net = tf.keras.Sequential(self.layers)

                self.y_pred = self.neural_net(self.tf_x)
                self.scale = tf.nn.softplus(
                    tf.Variable(tf.ones(self.y_pred.get_shape()))
                )
                self.y_sample = tfd.Normal(loc=self.y_pred, scale=self.scale)

            self.kl = 0
            for _, layer in enumerate(self.layers):
                self.kl += sum(layer.losses) / float(self.batch_size)
            self.reg_loss = -tf.reduce_sum(self.y_sample.log_prob(self.tf_y))
            self.loss = self.reg_loss + (self.reg * self.kl)

            self.optimizer = tf.compat.v1.train.AdamOptimizer(
                learning_rate=self.learning_rate
            )
            self.train_op = self.optimizer.minimize(self.loss)

            self.init_op = tf.group(
                tf.compat.v1.global_variables_initializer(),
                tf.compat.v1.local_variables_initializer(),
            )
            self.sess = tf.compat.v1.Session(graph=self.graph)
            with self.sess.as_default():
                self.sess.run(self.init_op)
                self.saver = tf.compat.v1.train.Saver()
