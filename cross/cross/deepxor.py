import os
import copy
import numpy as np

import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras import layers
from algorithm.parameters import params
from representation.derivation import legal_productions, generate_tree
from representation.tree import Tree
import brainfuck as bf

tf.flags.DEFINE_float("l2", default=1e-4, help="l2 regularization")
tf.flags.DEFINE_integer("input_layer", default=4096, help="input layer size")
tf.flags.DEFINE_integer("hidden_layer", default=2048, help="hidden layer size")
tf.flags.DEFINE_integer("policy_layer", default=512, help="policy layer size")
tf.flags.DEFINE_integer("value_layer", default=512, help="value layer size")
tf.flags.DEFINE_integer("seq_max_len", default=20, help="maximum state len")


FLAGS = tf.flags.FLAGS

# maximum number of syntax tree nodes
num_tree_nodes = 1
num_productions = 1
#num_choices = 25
solved = np.zeros([FLAGS.seq_max_len*num_tree_nodes*num_productions])
action_list = [ ]
len_solved = len(solved)
num_actions = 30 # FLAGS.seq_max_len #num_productions

validation = [
        [0 for x in range(0, len_solved)],
        ]

def apply_action(state, action):
    state = np.array([x for x in state])
    state[action[0]] = action[1]
    return state

def apply_action_reverse(state, action):
    return np.cross(action_list[action], state)

def _generate_tree(tree, output, selected_production, unexpanded):
    productions = params['BNF_GRAMMAR'].rules[tree.root]
    if selected_production == -1:
#        if len(productions['choices']) == 1:
#            selected_production = 0
#        else:
        unexpanded.append(tree)
        return output, unexpanded

    chosen_prod = productions['choices'][int(selected_production)]
    tree.children = []
#    print(chosen_prod)

    for symbol in chosen_prod['choice']:
        # Iterate over all symbols in the chosen production.
        if symbol["type"] == "T":
            # The symbol is a terminal. Append new node to children.
            tree.children.append(Tree(symbol["symbol"], tree, type=symbol["type"]))
            
            # Append the terminal to the output list.
            output.append(symbol["symbol"])
        
        elif symbol["type"] == "NT":
            # The symbol is a non-terminal. Append new node to children.
            tree.children.append(Tree(symbol["symbol"], tree, type=symbol["type"]))

            output, unexpanded = _generate_tree(tree.children[-1], output, -1, unexpanded)

    return output, unexpanded

def walk_tree(tree, output):
    for t in tree.children:
        walk_tree(t, output)
    output += tree.root if tree.type == "T" else []
    return output

def generate_output(state):
#    print(state)
    grm = params['BNF_GRAMMAR']
    ind_tree = Tree(str(grm.start_rule["symbol"]), None)
    unexpanded = []
    output = []
    tree = ind_tree
    s = copy.copy(state)
    idx = 0
    while True:
        output, unexpanded = _generate_tree(tree, output, s[idx], unexpanded)
        idx += 1
        tree = unexpanded.pop(0) if unexpanded else None
        if tree is None or s[idx] == -1:
            break
    return walk_tree(ind_tree, []), ind_tree


def reward(state):

    output, _ = generate_output(state)
#    print(output)

    with open('/tmp/program.bf', 'w') as out:
        out.write("".join(output))

    retval = os.system("./eval.sh")

#    code="".join(output)
#    retval = bf.bf(code,0, len(code)-1, [0 for x in range(0,9999)], 0, max_steps=100)

#    print(retval)
#    return retval
    if retval == 0:
        return 1
    else:
        return -1

def state_diff(state):
    return np.linalg.norm([_solved - _state for _solved, _state in zip(solved, state)])

class Position():
    def __init__(self, n=0, state=None):
        self.n = n
        self.state = -1*np.ones([FLAGS.seq_max_len*num_tree_nodes*num_productions]) if state is None else state
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

