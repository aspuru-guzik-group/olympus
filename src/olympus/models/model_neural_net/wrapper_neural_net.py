#!/usr/bin/env python

from olympus.models import WrapperTensorflowModel

#===============================================================================

import numpy as np
import time
import tensorflow as tf

#===============================================================================

class NeuralNet(WrapperTensorflowModel):

    ATT_KIND = {'type': 'string', 'default': 'NeuralNet'}

    def __init__(self, scope='model', hidden_depth=3, hidden_nodes=48, hidden_act='leaky_relu', out_act='linear',
                 l2_activity=1e-3, gaussian_dropout=0.0, dropout=0.1,
                 learning_rate=1e-3, pred_int=100,  reg=0.001, es_patience=100, max_epochs=100000, batch_size=20):
        '''Neural network model.

        Args:
            scope (str): TensorFlow scope.
            hidden_depth (int): Number of hidden layers.
            hidden_nodes (int): Number of hidden nodes per layer.
            hidden_act (str): Hidden activation function. Available options are 'linear', 'leaky_relu', 'relu',
                'softmax', 'softplus', 'softsign', 'sigmoid'.
            out_act (str): Output activation function. Available options are 'linear', 'leaky_relu', 'relu',
                'softmax', 'softplus', 'softsign', 'sigmoid'.
            l2_activity (float): L2 regularization.
            learning_rate (float): Learning rate.
            pred_int (int): Frequency with which we make predictions on the validation/training set (in number of epochs).
            reg (float): ???
            es_patience (int): Early stopping patience.
            max_epochs (int): Maximum number of epochs allowed.
            batch_size (int): Size batches used for training.
        '''

        WrapperTensorflowModel.__init__(**locals())

    def _build_inference(self):

        self.graph = tf.Graph()
        self.is_graph_constructed = True

        activation_hidden = self.act_funcs(self.hidden_act)
        activation_out    = self.act_funcs(self.out_act)

        STDDEV = 1e-1

        with self.graph.as_default():
            self.tf_x = tf.compat.v1.placeholder(tf.float32, [self.batch_size, self.features_dim])
            self.tf_y = tf.compat.v1.placeholder(tf.float32, [self.batch_size, self.targets_dim])

            with tf.name_scope(self.scope):

                self.layers = []
                for _ in range(self.hidden_depth):
                    self.layers.append(tf.keras.layers.BatchNormalization())
                    self.layers.append(tf.keras.layers.Dense(self.hidden_nodes,
                        activation=activation_hidden,
                        kernel_regularizer=tf.keras.regularizers.l2(l = self.l2_activity)))
                    self.layers.append(tf.keras.layers.GaussianDropout(self.gaussian_dropout))
                    self.layers.append(tf.keras.layers.Dropout(self.dropout))

                self.layers.append(tf.keras.layers.BatchNormalization())
                self.layers.append(tf.keras.layers.Dense(self.targets_dim,
                                               activation = activation_out,
                                               kernel_regularizer = tf.keras.regularizers.l2(l = self.l2_activity)))

                self.neural_net = tf.keras.Sequential(self.layers)
                self.y_pred     = self.neural_net(self.tf_x)

                reg_vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.REGULARIZATION_LOSSES)
                self.reg_loss   = self.reg * tf.reduce_mean(reg_vars)
                self.pred_loss  = tf.sqrt(tf.reduce_mean(tf.square(self.y_pred - self.tf_y)))
                self.pred_loss += tf.reduce_mean(tf.abs(self.y_pred - self.tf_y))
                self.loss       = self.pred_loss + self.reg_loss

                self.optimizer = tf.compat.v1.train.AdamOptimizer(self.learning_rate)
                self.train_op  = self.optimizer.minimize(self.loss)

                self.init_op   = tf.group(tf.compat.v1.global_variables_initializer(), tf.compat.v1.local_variables_initializer())
                self.sess = tf.compat.v1.Session(graph = self.graph)
                with self.sess.as_default():
                    self.sess.run(self.init_op)
                    self.saver = tf.compat.v1.train.Saver()

#===============================================================================
