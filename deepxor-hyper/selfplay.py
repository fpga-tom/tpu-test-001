import tensorflow as tf
from strategies import MCTSPlayer
import dual_net
from deepxor import state_diff
import time

#tf.flags.DEFINE_string("model_dir", default=None, help="Estimator model dir")
tf.flags.DEFINE_integer("time_per_move", default=5, help="Thinking time per move")

FLAGS = tf.flags.FLAGS
import sys
sys.setrecursionlimit(5000)

def play(network):
    readouts = FLAGS.num_readouts
    player = MCTSPlayer(network)
    player.initialize_game()

    first_node = player.root.select_leaf()
    prob, val = network.predict(first_node.position.state)
    first_node.incorporate_results(prob, val, first_node)

    lastmove = -1
    hamm_dist = state_diff(player.root.position.state)

    for lo in range(0, hamm_dist):
#        player.root.inject_noise()
        current_readouts = player.root.N
        start = time.time()
        while player.root.N < current_readouts + readouts and time.time() - start < FLAGS.time_per_move:
            player.tree_search()

        move = player.pick_move()
        if move == lastmove:
            tf.logging.info('lastmove == move')
            return state_diff(player.root.position.state)
        before = state_diff(player.root.position.state)
        player.play_move(move)
        after = state_diff(player.root.position.state)
        if after > before:
            tf.logging.info('move increasing distance')
            return after
        tf.logging.info('playing move: %d hamming distance: %d' % (move, state_diff(player.root.position.state)))
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
