import tensorflow as tf
from cross.strategies import MCTSPlayer
import cross.dual_net
from cross.deepxor import state_diff
from cross.utilities import PositionFactory
import time
import numpy as np

#tf.flags.DEFINE_string("model_dir", default=None, help="Estimator model dir")
tf.flags.DEFINE_integer("time_per_move", default=5, help="Thinking time per move")

FLAGS = tf.flags.FLAGS
import sys
sys.setrecursionlimit(5000)

g_network = None
g_player = None

def play_init(network, state=None, tree=None):
    global g_network
    global g_player
    g_network = network
    player = MCTSPlayer(network)
    player.initialize_game(PositionFactory.factory().create(state=state, trees=[tree]))
    g_player = player

    first_node = player.root.select_leaf()
    prob, val = network.predict(first_node.position.state)
    first_node.incorporate_results(prob, val, first_node)

    return player

def play_select_move():
    global g_network
    global g_player
    readouts = FLAGS.num_readouts
    current_readouts = g_player.root.N
    start = time.time()
    while g_player.root.N < current_readouts + readouts and time.time() - start < FLAGS.time_per_move:
        g_player.tree_search()
    move = g_player.pick_move()
    if g_player.root.position._output is not None:
        print("".join(g_player.root.position._output))
        r = g_player.root.position.current
        while r is not None and r.parent is not None:
            print(r)
            r = r.parent
    g_player.play_move(move)


    return move

def play(network, state=None):
    readouts = FLAGS.num_readouts
    player = play_init(network, state)

    lastmove = -1
#    hamm_dist = state_diff(player.root.position.state)
    hamm_dist = 25

    for lo in range(0, hamm_dist):
#        player.root.inject_noise()
        current_readouts = player.root.N
        start = time.time()
        while player.root.N < current_readouts + readouts and time.time() - start < FLAGS.time_per_move:
            player.tree_search()

        move = player.pick_move()
        if move == lastmove:
            tf.logging.info('lastmove == move')
#            return state_diff(player.root.position.state)
        before = state_diff(player.root.position.state)
        player.play_move(move)
        after = state_diff(player.root.position.state)
        if after > before:
            tf.logging.info('move increasing distance')
            return after
        if after < 0.1:
            tf.logging.info('done')
            return after
        tf.logging.info('playing move: %d euclidean distance: %f' % (move, state_diff(player.root.position.state)))
        if player.root.is_done():
            tf.logging.info('done')
            return 0
        lastmove = move
    return state_diff(player.root.position.state)


def main(argv):
    network = dual_net.Network()

    play(network)

if __name__ == "__main__":
    tf.app.run()
