from representation.grammar import Grammar
from representation.tree import Tree
from representation.derivation import generate_tree
from algorithm.parameters import params
from representation.individual import Individual
from representation.position import PonyGEPositionFactory
from cross.utilities import PositionFactory
from cross.selfplay import play_init, play_select_move
from cross.dual_net import Network

import tensorflow as tf

method="random"
max_depth=40
PositionFactory.set_factory('pony', PonyGEPositionFactory(method=method,max_depth=max_depth))

grm = Grammar('./tbasic.bnf')
params['BNF_GRAMMAR'] = grm
ind_tree = Tree(str(grm.start_rule["symbol"]), None)
play_init(Network(), tree=ind_tree)

for i in range(0,max_depth):
    move = play_select_move()
    print(move)
    #print(ind_tree)


#genome, output, nodes, _, depth = generate_tree(ind_tree, [], [], method, 0, 0, 0, max_depth)
#ind = Individual(genome, ind_tree, map_ind=False)
#ind.phenotype = "".join(output)
#print(genome)
#print(ind.phenotype)

