import copy
import numpy as np

import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras import layers

#solved = [1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1]
solved = [1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0]
len_solved = len(solved)
num_actions = len_solved

def apply_action(state, action):
    state = [i for i in state]
    if action < len(solved):
        state[action] ^= 1
    return state

def reward(state):
    if all([_solved == _state for _solved, _state in zip(solved, state)]):
        return 1
    return -1

def state_diff(state):
    return sum([1 for _solved, _state in zip(solved, state) if _solved != _state])

class Position():
    def __init__(self, n=0):
        self.n = n
        self.state = [0 for x in range(0, len_solved)]
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
        self.l_0 = layers.Dense(4096, activation=tf.nn.elu)
        self.l_1 = layers.Dense(2048, activation=tf.nn.elu)
        self.l_2 = layers.Dense(512, activation=tf.nn.elu)
        self.l_3 = layers.Dense(512, activation=tf.nn.elu)
        self.logits = layers.Dense(num_actions)
        self.policy = layers.Softmax()
        self.values = layers.Dense(1, activation=tf.tanh)

    def __call__(self, input_layer):
#        with tf.variable_scope(self.scope):
#            l_0 = tf.layers.Dense(input_layer, 4096, activation=tf.nn.elu, name='l0')
#            l_1 = tf.layers.dense(l_0, 2048, activation=tf.nn.elu, name='l1')
#            l_2 = tf.layers.dense(l_1, 512, activation=tf.nn.elu, name='l2')
#            l_3 = tf.layers.dense(l_1, 512, activation=tf.nn.elu, name='l3')
#            logits = tf.layers.dense(l_2, num_actions, name='logits')
#            policy_output = tf.nn.softmax(logits, name='policy')
#            value_output = tf.layers.dense(l_3, 1, activation=tf.tanh, name='value')
#
#            sub_layers = [l_0, l_1, l_2, l_3, logits, policy_output, value_output]
#            self.weights = []
#            for layer in sub_layers:
#                self.weights += layer.trainable_weights
#
#            return policy_output, value_output, logits

        l0 = self.l_0(input_layer)
        l1 = self.l_1(l0)
        l2 = self.l_2(l1)
        l3 = self.l_3(l1)
        _logits = self.logits(l2)
        _policy = self.policy(_logits)
        _values = self.values(l3)

        return _policy, _values, _logits

