import tensorflow as tf

tf.flags.DEFINE_string("grammar", default="./tbasic.bnf", help="grammar")
from representation.grammar import Grammar
from representation.tree import Tree
from representation.derivation import generate_tree
from algorithm.parameters import params
from representation.individual import Individual
from representation.position import PonyGEPositionFactory
from representation.recurrent import RecurrentModelFactory
from cross.utilities import PositionFactory, ModelFactory
from cross.selfplay import play_init, play_select_move
from cross.dual_net import Network
from cross.deepxor import num_actions

FLAGS = tf.flags.FLAGS

method="random"
max_depth=100
PositionFactory.set_factory('pony', PonyGEPositionFactory(method=method,max_depth=max_depth))
ModelFactory.set_factory('recurrent', RecurrentModelFactory(num_actions))

grm = Grammar(FLAGS.grammar)
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

