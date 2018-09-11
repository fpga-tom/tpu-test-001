"""
Implements Monte Carlo Tree search
"""

import collections
import math

from absl import flags
import numpy as np
import deepxor

flags.DEFINE_float('c_puct', 0.96,
                   'Exploration constant balancing priors vs. value net output.')

flags.DEFINE_float('dirichlet_noise_alpha', 0.03 * 361 / (deepxor.num_actions ** 2),
                   'Concentrated-ness of the noise being injected into priors.')
flags.register_validator('dirichlet_noise_alpha', lambda x: 0 <= x < 1)

FLAGS = flags.FLAGS

class DummyNode(object):

    def __init__(self):
        self.parent = None
        self.child_N = collections.defaultdict(float)
        self.child_W = collections.defaultdict(float)

class MCTSNode(object):

    def __init__(self, position, fmove=None, parent=None):
        if parent is None:
            parent = DummyNode()
        else:
            assert fmove is not None
        self.parent = parent
        self.fmove = fmove
        self.position = position
        self.is_expanded = False
        self.losses_applied = 0
        self.illegal_moves = 1 - self.position.all_legal_moves()
        self.child_N = np.zeros([deepxor.num_actions], dtype=np.float32)
        self.child_W = np.zeros([deepxor.num_actions], dtype=np.float32)
        self.original_prior = np.zeros([deepxor.num_actions], dtype=np.float32)
        self.child_prior = np.zeros([deepxor.num_actions], dtype=np.float32)
        self.children = {}

    @property
    def child_action_score(self):
        return (self.child_Q * self.position.to_play +
                self.child_U - 1000 * self.illegal_moves)

    @property
    def child_Q(self):
        return self.child_W / (1 + self.child_N)

    @property
    def child_U(self):
        return (FLAGS.c_puct * math.sqrt(1 + self.N) *
                self.child_prior / (1 + self.child_N))

    @property
    def Q(self):
        return self.W / (1 + self.N)

    @property
    def N(self):
        return self.parent.child_N[self.fmove]

    @N.setter
    def N(self, value):
        self.parent.child_N[self.fmove] = value

    @property
    def W(self):
        return self.parent.child_W[self.fmove]

    @W.setter
    def W(self, value):
        self.parent.child_W[self.fmove] = value

    def select_leaf(self):
        current = self
        while True:
            current.N += 1
            if not current.is_expanded:
                break
            best_move = np.argmax(current.child_action_score)
            current = current.maybe_add_child(best_move)
        return current

    def maybe_add_child(self, fcoord):
        if fcoord not in self.children:
            new_position = self.position.play_move(fcoord)
            self.children[fcoord] = MCTSNode(
                    new_position, fmove=fcoord, parent=self)
        return self.children[fcoord]

    def add_virtual_loss(self, up_to):
        self.losses_applied += 1
        loss = self.position.to_play
        self.W += loss
        if self.parent is None or self is up_to:
            return
        self.parent.add_virtual_loss(up_to)

    def revert_virtual_loss(self, up_to):
        self.losses_applied -= 1
        revert = -1 * self.position.to_play
        self.W += revert
        if self.parent is None or self is up_to:
            return
        self.parent.revert_virtual_loss(up_to)

    def revert_visits(self, up_to):
        self.N -= 1
        if self.parent is None or self is up_to:
            return
        self.parent.revert_visits(up_to)

    def incorporate_results(self, move_probabilities, value, up_to):
        if self.is_expanded:
            self.revert_visits(up_to=up_to)
            return
        self.is_expanded = True
        move_probs = move_probabilities * (1 - self.illegal_moves)
        scale = sum(move_probs)
        if scale > 0:
            move_probs *= 1 / scale

        self.original_prior = self.child_prior = move_probs
        self.child_W = np.ones([deepxor.num_actions], dtype=np.float32) * value
        self.backup_value(value, up_to=up_to)

    def backup_value(self, value, up_to):
        self.W += value
        if self.parent is None or self is up_to:
            return
        self.parent.backup_value(value, up_to)

    def children_as_pi(self, squash=False):
        probs = self.child_N
        if squash:
            probs = probs ** .98
        sum_probs = np.sum(probs)
        if sum_probs == 0:
            return probs
        return probs / np.sum(probs)

    def inject_noise(self):
        epsilon = 1e-5
        legal_moves = (1 - self.illegal_moves) + epsilon
        a = legal_moves * ([FLAGS.dirichlet_noise_alpha] * (deepxor.num_actions))
        dirichlet = np.random.dirichlet(a)
        self.child_prior = (self.child_prior * (1 - FLAGS.dirichlet_noise_weight) +
                            dirichlet * FLAGS.dirichlet_noise_weight)

    def is_done(self):
        return self.position.score() == 1
