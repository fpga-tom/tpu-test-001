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

    def set_tree(self, tree):
        self.position.current = tree
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
    def construct(n, state, tree=None, method=None, max_depth=0):
        return PonyGEPositionBuilder(n, state).set_tree(tree).set_method(method).set_max_depth(max_depth).get_result()

class PonyGEPositionFactory(PositionFactory):
    def __init__(self, method, max_depth):
        self.method = method
        self.max_depth = max_depth

    def create(self, n=0, state=None, tree=None):
        return PonyGEPositionBuilderDirector.construct(n, state, tree=tree, method=self.method, max_depth=self.max_depth)

class PonyGEPosition(cross.deepxor.Position):

    def __init__(self, n=0, state=None, trees=None, method=None):
        super(PonyGEPosition, self).__init__(n=n, state=state)
        self.method = method
        self.max_depth = None
        self.genome = []
        self.output = []
        self._output = None
        self.nodes = 0
        self.max_depth = 20
        self.depth = 0
        self.depth_limit = 20
        self.current = None
        self.num_nodes = 0
        self.undecided_trees = []

    def _generate_tree(self, pos, tree, output, selected_production, undecided_trees):
        productions = params['BNF_GRAMMAR'].rules[tree.root]
        if selected_production == -1:
#            if  len(productions['choices']) == 1:
#                selected_production = 0
#            else:
             undecided_trees.append(tree)
             return output, undecided_trees

        chosen_prod = productions['choices'][selected_production]
        tree.children = []
        pos.state = apply_action(pos.state, (pos.num_nodes , selected_production))
        pos.num_nodes += 1

        for symbol in chosen_prod['choice']:
            # Iterate over all symbols in the chosen production.
            if symbol["type"] == "T":
                # The symbol is a terminal. Append new node to children.
                tree.children.append(Tree(symbol["symbol"], tree))
                
                # Append the terminal to the output list.
                output.append(symbol["symbol"])
            
            elif symbol["type"] == "NT":
                # The symbol is a non-terminal. Append new node to children.
                tree.children.append(Tree(symbol["symbol"], tree))

                output, undecided_trees = self._generate_tree(pos, tree.children[-1], output, -1, undecided_trees)

        return output, undecided_trees

    def play_move(self, c):

        pos = copy.deepcopy(self)

        if pos.current:
            pos.output, pos.undecided_trees = self._generate_tree(pos, pos.current, pos.output, c, pos.undecided_trees)
            pos.current = pos.undecided_trees.pop(0) if pos.undecided_trees else None

        pos.n += 1
        return pos

    def is_done(self):
        return self.current is None


    def all_legal_moves(self):
        available_indices = np.zeros([num_actions])
        if self.current and self.current.root in params['BNF_GRAMMAR'].rules:
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
