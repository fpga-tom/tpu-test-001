import copy
import numpy as np

import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras import layers

tf.flags.DEFINE_float("l2", default=1e-4, help="l2 regularization")
tf.flags.DEFINE_integer("input_layer", default=4096, help="input layer size")
tf.flags.DEFINE_integer("hidden_layer", default=2048, help="hidden layer size")
tf.flags.DEFINE_integer("policy_layer", default=512, help="policy layer size")
tf.flags.DEFINE_integer("value_layer", default=512, help="value layer size")


FLAGS = tf.flags.FLAGS

solved = [10, 7, 12]
action_list = [
        [1,0,0],
        [0,1,0],
        [0,0,1],
        [1,1,1],
        [2,0,0],
        [0,2,0],
        [0,0,2],
        [2,2,2],
        [3,0,0],
        [0,3,0],
        [0,0,3],
        [3,3,3],
        [4,0,0],
        [0,4,0],
        [0,0,4],
        [4,4,4]
        ]
#solved = [1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0]
#MODEL_DIR=/home/tomas/Documents/deepxor-async-gd-half-1
len_solved = len(solved)
num_actions = len(action_list)

validation = [
        [0 for x in range(0, len_solved)],
        ]

def apply_action(state, action):
    state = [i + a for i,a in zip(state, action_list[action])]
    return state

def apply_action_reverse(state, action):
    state = [i - a for i,a in zip(state, action_list[action])]
    return state

def reward(state):
    if all([_solved == _state for _solved, _state in zip(solved, state)]):
        return 1
    return -1

def state_diff(state):
    return np.linalg.norm([_solved - _state for _solved, _state in zip(solved, state)])

class Position():
    def __init__(self, n=0, state=None):
        self.n = n
        self.state = [0 for x in range(0, len_solved)] if state is None else state
        self.to_play = 1

    def play_move(self, c):

        pos = copy.deepcopy(self)
        pos.state = apply_action(pos.state, c)
        pos.n += 1

        return pos

    def score(self):
        return reward(self.state)

    def all_legal_moves(self):
        return np.ones([num_actions])


class DeepxorModel(keras.Model):
    def __init__(self, scope):
        self.scope = scope
        super(DeepxorModel, self).__init__()
        self.l_0 = layers.Dense(FLAGS.input_layer, activation=tf.nn.elu, kernel_regularizer= tf.contrib.layers.l2_regularizer(scale=FLAGS.l2), kernel_initializer=tf.contrib.layers.xavier_initializer())
        self.l_1 = layers.Dense(FLAGS.hidden_layer, activation=tf.nn.elu, kernel_regularizer= tf.contrib.layers.l2_regularizer(scale=FLAGS.l2), kernel_initializer=tf.contrib.layers.xavier_initializer())
        self.l_2 = layers.Dense(FLAGS.policy_layer, activation=tf.nn.elu, kernel_regularizer= tf.contrib.layers.l2_regularizer(scale=FLAGS.l2), kernel_initializer=tf.contrib.layers.xavier_initializer())
        self.l_3 = layers.Dense(FLAGS.value_layer, activation=tf.nn.elu, kernel_regularizer= tf.contrib.layers.l2_regularizer(scale=FLAGS.l2), kernel_initializer=tf.contrib.layers.xavier_initializer())
        self.logits = layers.Dense(num_actions,kernel_regularizer= tf.contrib.layers.l2_regularizer(scale=FLAGS.l2), kernel_initializer=tf.contrib.layers.xavier_initializer())
        self.policy = layers.Softmax()
        self.values = layers.Dense(1, activation=tf.tanh,kernel_regularizer= tf.contrib.layers.l2_regularizer(scale=FLAGS.l2), kernel_initializer=tf.contrib.layers.xavier_initializer())

    def __call__(self, input_layer):

        l0 = self.l_0(input_layer)
        l1 = self.l_1(l0)
        l2 = self.l_2(l1)
        l3 = self.l_3(l1)
        _logits = self.logits(l2)
        _policy = self.policy(_logits)
        _values = self.values(l3)

        return _policy, _values, _logits

