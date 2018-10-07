import copy
import numpy as np

import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras import layers
from cross.utilities import ModelFactory

tf.flags.DEFINE_integer("lstm_size", default=100, help="lstm layer size")

FLAGS = tf.flags.FLAGS

class RecurrentModelFactory(ModelFactory):

    def __init__(self, num_actions):
        self.num_actions = num_actions

    def create(self):
        return RecurrentModel(self.num_actions)


class RecurrentModel(keras.Model):
    def __init__(self, num_actions):
        super(RecurrentModel, self).__init__()
        self.l_0 = layers.Dense(FLAGS.input_layer, activation=tf.nn.elu, kernel_regularizer= tf.contrib.layers.l2_regularizer(scale=FLAGS.l2), kernel_initializer=tf.contrib.layers.xavier_initializer())
        self.l_1 = layers.Dense(FLAGS.hidden_layer, activation=tf.nn.elu, kernel_regularizer= tf.contrib.layers.l2_regularizer(scale=FLAGS.l2), kernel_initializer=tf.contrib.layers.xavier_initializer())
        self.l_2 = layers.Dense(FLAGS.policy_layer, activation=tf.nn.elu, kernel_regularizer= tf.contrib.layers.l2_regularizer(scale=FLAGS.l2), kernel_initializer=tf.contrib.layers.xavier_initializer())
        self.l_3 = layers.Dense(FLAGS.value_layer, activation=tf.nn.elu, kernel_regularizer= tf.contrib.layers.l2_regularizer(scale=FLAGS.l2), kernel_initializer=tf.contrib.layers.xavier_initializer())
        self.logits = layers.Dense(num_actions,kernel_regularizer= tf.contrib.layers.l2_regularizer(scale=FLAGS.l2), kernel_initializer=tf.contrib.layers.xavier_initializer())
        self.policy = layers.Softmax()
        self.values = layers.Dense(1, activation=tf.tanh,kernel_regularizer= tf.contrib.layers.l2_regularizer(scale=FLAGS.l2), kernel_initializer=tf.contrib.layers.xavier_initializer())

        self.seqlen = tf.placeholder(tf.int32,[None])

    def dynamicRNN(self, x, seqlen):

        # Prepare data shape to match `rnn` function requirements
        # Current data input shape: (batch_size, n_steps, n_input)
        # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)
        
        # Unstack to get a list of 'n_steps' tensors of shape (batch_size, n_input)
        x = tf.unstack(x, FLAGS.seq_max_len, 1)

        # Define a lstm cell with tensorflow
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(FLAGS.lstm_size)

        # Get lstm cell output, providing 'sequence_length' will perform dynamic
        # calculation.
        outputs, states = tf.contrib.rnn.static_rnn(lstm_cell, x, dtype=tf.float32,
                                    sequence_length=seqlen)

        # When performing dynamic calculation, we must retrieve the last
        # dynamically computed output, i.e., if a sequence length is 10, we need
        # to retrieve the 10th output.
        # However TensorFlow doesn't support advanced indexing yet, so we build
        # a custom op that for each sample in batch size, get its length and
        # get the corresponding relevant output.

        # 'outputs' is a list of output at every timestep, we pack them in a Tensor
        # and change back dimension to [batch_size, n_step, n_input]
        outputs = tf.stack(outputs)
        outputs = tf.transpose(outputs, [1, 0, 2])

        # Hack to build the indexing and retrieve the right output.
        batch_size = tf.shape(outputs)[0]
        # Start indices for each sample
        index = tf.range(0, batch_size) * seq_max_len + (seqlen - 1)
        # Indexing
        outputs = tf.gather(tf.reshape(outputs, [-1, FLAGS.lstm_size]), index)

        return outputs

    def __call__(self, input_layer):
        
#        rnn = self.dynamicRNN(input_layer, self.seqlen)

        l0 = self.l_0(input_layer)
        l1 = self.l_1(l0)
        l2 = self.l_2(l1)
        l3 = self.l_3(l1)
        _logits = self.logits(l2)
        _policy = self.policy(_logits)
        _values = self.values(l3)

        return _policy, _values, _logits

