"""
Implements MCTS tree search
"""

import random
import deepxor
import numpy as np
from absl import flags
import mcts

flags.DEFINE_integer('softpick_move_cutoff', 2, 'The move number (<) up to which are moves softpicked from MCTS visits.')
flags.DEFINE_integer('parallel_readouts', 8, 'Number of searches to execute in parallel. Also neural network batch size')
flags.DEFINE_integer('num_readouts', 800, 'Number of searches to add to the MCTS search tree before playing a move')

FLAGS = flags.FLAGS

class MCTSPlayer(object):

    def __init__(self, network):
        self.root = None
        self.initialize_game()
        self.temp_threshold = FLAGS.softpick_move_cutoff
        self.network = network

    def initialize_game(self, position=None):
        if position is None:
            position = deepxor.Position()
        self.root = mcts.MCTSNode(position)

    def play_move(self, move):
        self.root = self.root.maybe_add_child(move)
        self.position = self.root.position
        del self.root.parent.children

    def pick_move(self):
        '''Picks a move to play, based on MCTS readout statistics.

        Highest N is most robust indicator. In the early stage of the game, pick
        a move weighted by visit count; later on, pick the absolute max.'''
        '''
        if self.root.position.n >= self.temp_threshold:
            fcoord = np.argmax(self.root.child_N)
            print('fcooord 1', fcoord)
        else:
            cdf = self.root.children_as_pi(squash=True).cumsum()
            selection = random.random()
            fcoord = cdf.searchsorted(selection)
            '''
        fcoord = np.argmax(self.root.child_N)
        return fcoord

    def tree_search(self, parallel_readouts=None):
        if parallel_readouts is None:
            parallel_readouts = FLAGS.parallel_readouts
        leaves = []
        while len(leaves) < parallel_readouts:
            leaf = self.root.select_leaf()
            if leaf.is_done():
                value = 1 if leaf.position.score() > 0 else -1
                leaf.backup_value(value, up_to=self.root)
                continue
            leaf.add_virtual_loss(up_to=self.root)
            leaves.append(leaf)
        if leaves:
            move_probs, values = self.network.run_many(
                    [leaf.position for leaf in leaves])
            for leaf, move_prob, value in zip(leaves, move_probs, values):
                leaf.revert_virtual_loss(up_to=self.root)
                leaf.incorporate_results(move_prob, value, up_to=self.root)
        return leaves
