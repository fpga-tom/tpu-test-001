import copy
import numpy as np
import cross.deepxor

from abc import ABCMeta, abstractmethod
from cross.utilities import PositionFactory
from cross.deepxor import num_actions
from algorithm.parameters import params
from representation.derivation import legal_productions, generate_tree

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
        self.position.tree = tree
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

    def __init__(self, n=0, state=None, tree=None, method=None):
        super(PonyGEPosition, self).__init__(n=n, state=state)
        self.tree = tree
        self.method = method
        self.max_depth = None
        self.genome = []
        self.output = []
        self._output = None
        self.nodes = 0
        self.max_depth = 0
        self.depth = 0
        self.depth_limit = 90
        self.kind = "NT"

    def play_move(self, c):
        tree = copy.copy(self.tree)
        genome = copy.copy(self.genome)
        output = copy.copy(self.output)
        nodes = copy.copy(self.nodes)
        depth = self.depth
        max_depth = self.max_depth
        method = self.method

        if self.kind == "NT":
            _genome, _output, _nodes, _depth, _max_depth = generate_tree(tree, genome, output, method, nodes, depth, max_depth, self.depth_limit, c)
        
            self._output = _output

            if len(tree.children) > 1:
                position_list = []
                for t in tree.children:
                    result = super(PonyGEPosition, self).play_move(c)
                    result.genome = genome + _genome
                    result.output = output + _output
                    result.nodes = _nodes
                    result.depth = _depth
                    result.max_depth = _max_depth
                    result.tree = t
                    result.kind = "NT" if t.root in params['BNF_GRAMMAR'].rules else "T"
                    position_list.append(result)
                merged = PonyGEMergedPosition(position_list=position_list)
                return merged
            elif len(tree.children) == 1:
                t = tree.children[-1]
                result = super(PonyGEPosition, self).play_move(c)
                result.genome = genome + _genome
                result.output = output + _output
                result.nodes = _nodes
                result.depth = _depth
                result.max_depth = _max_depth
                result.tree = t
                result.kind = "NT" if t.root in params['BNF_GRAMMAR'].rules else "T"
                return result
            raise Exception()
        else:
                result = super(PonyGEPosition, self).play_move(c)
                result.kind = "T"
                return result



    def all_legal_moves(self):
        available_indices = np.zeros([num_actions])
        if self.tree.root in params['BNF_GRAMMAR'].rules:
            productions = params['BNF_GRAMMAR'].rules[self.tree.root]
            remaining_depth = self.max_depth - self.n
            available = legal_productions(self.method, remaining_depth, self.tree.root,
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
