import copy
import numpy as np
import cross.deepxor
import sys

from abc import ABCMeta, abstractmethod
from cross.utilities import PositionFactory
from cross.deepxor import num_actions, apply_action, num_tree_nodes, num_productions
from algorithm.parameters import params
from representation.derivation import legal_productions, generate_tree
from representation.tree import Tree

class Builder:
    __metaclass__ = ABCMeta

    @abstractmethod
    def set_tree(self, tree):
        pass

    @abstractmethod
    def set_method(self, method):
        pass

class PonyGEPositionBuilder(Builder):
    def __init__(self, n, state):
        self.position = PonyGEPosition(n, state)

    def set_trees(self, trees):
        self.position.trees = trees
        return self

    def set_method(self, method):
        self.position.method = method
        return self

    def set_max_depth(self, max_depth):
        self.position.max_depth = max_depth
        return self

    def get_result(self):
        return self.position

class PonyGEPositionBuilderDirector:
    @staticmethod
    def construct(n, state, trees=None, method=None, max_depth=0):
        return PonyGEPositionBuilder(n, state).set_trees(trees).set_method(method).set_max_depth(max_depth).get_result()

class PonyGEPositionFactory(PositionFactory):
    def __init__(self, method, max_depth):
        self.method = method
        self.max_depth = max_depth

    def create(self, n=0, state=None, trees=None):
        return PonyGEPositionBuilderDirector.construct(n, state, trees=trees, method=self.method, max_depth=self.max_depth)

class PonyGEPosition(cross.deepxor.Position):

    def __init__(self, n=0, state=None, trees=None, method=None):
        super(PonyGEPosition, self).__init__(n=n, state=state)
        self.trees = trees if trees is not None else []
        self.method = method
        self.max_depth = None
        self.genome = []
        self.output = []
        self._output = None
        self.nodes = 0
        self.max_depth = 0
        self.depth = 0
        self.depth_limit = 90
        self.current = None

    def play_move(self, c):

        pos = copy.deepcopy(self)
        pos.current = None
        if self.current is None:
            pos.current = copy.deepcopy(self.trees[c])
        else:
            tree = self.current
            productions = params['BNF_GRAMMAR'].rules[self.current.root]
            chosen_prod = productions['choices'][c]
            tree.children = []

            for symbol in chosen_prod['choice']:
                # Iterate over all symbols in the chosen production.
                if symbol["type"] == "T":
                    # The symbol is a terminal. Append new node to children.
                    tree.children.append(Tree(symbol["symbol"], tree))
                    
                    # Append the terminal to the output list.
                    self.output.append(symbol["symbol"])
                
                elif symbol["type"] == "NT":
                    # The symbol is a non-terminal. Append new node to children.
                    tree.children.append(Tree(symbol["symbol"], tree))
                    pos.trees.append(tree.children[-1])

                    idx = [k for k, v in params['BNF_GRAMMAR'].rules.items()].index(symbol["symbol"])
                    pos.state = apply_action(pos.state, pos.n * num_tree_nodes * num_productions + pos.n * num_productions + idx)

            self._output = self.output

        pos.n += 1
        return pos


    def all_legal_moves(self):
        available_indices = np.zeros([num_actions])
        if self.current is None:
            for i, tree in enumerate(self.trees):
                available_indices[i] = 1.0
        else:
            if self.current.root in params['BNF_GRAMMAR'].rules:
                productions = params['BNF_GRAMMAR'].rules[self.current.root]
                remaining_depth = self.max_depth - self.n
                available = legal_productions(self.method, remaining_depth, self.current.root,
                                              productions['choices'])
                for chosen_prod in available:
                    idx = productions['choices'].index(chosen_prod) 
                    available_indices[idx] = 1.
        return available_indices


class PonyGEMergedPosition(PonyGEPosition):

    def __init__(self, n=0, state=None, tree=None, method=None, position_list=[]):
        super(PonyGEMergedPosition, self).__init__(n=n, state=state)
        self.position_list = position_list
        self.position_map = {}
        self._output = []

    def play_move(self, move):
        result = self.position_map[move].play_move(move)
        return result

    def all_legal_moves(self):
        available_indices = np.zeros([num_actions])
        for pos in self.position_list:
            if pos.tree.root in params['BNF_GRAMMAR'].rules:
                productions = params['BNF_GRAMMAR'].rules[pos.tree.root]
                remaining_depth = pos.max_depth - pos.n
                available = legal_productions(pos.method, remaining_depth, pos.tree.root,
                                              productions['choices'])
                for chosen_prod in available:
                    idx = productions['choices'].index(chosen_prod)
                    self.position_map[idx] = pos
                    available_indices[idx] = 1.
        return available_indices
