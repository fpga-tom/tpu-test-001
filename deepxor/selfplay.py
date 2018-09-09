import tensorflow as tf
from strategies import MCTSPlayer
import dual_net
from deepxor import state_diff

tf.flags.DEFINE_string("tpu", default=None, help="TPU which to use")
tf.flags.DEFINE_string("tpu_zone", default=None, help="GCE zone of TPU" )
tf.flags.DEFINE_string("gcp_project", default=None, help="Project name of TPU enabled project")

tf.flags.DEFINE_bool("use_tpu", default=False, help="Use TPU rather than CPU")
tf.flags.DEFINE_string("model_dir", default=None, help="Estimator model dir")
tf.flags.DEFINE_integer("batch_size", default=8, help="Batch size")
tf.flags.DEFINE_integer("iterations", default=50, help="Number of iterations per TPU loop")
tf.flags.DEFINE_integer("num_shards", default=8, help="Number of shards (TPU chips)")

FLAGS = tf.flags.FLAGS

def play(network):
    readouts = FLAGS.num_readouts
    player = MCTSPlayer(network)
    player.initialize_game()

    first_node = player.root.select_leaf()
    prob, val = network.predict(first_node.position.state)
    first_node.incorporate_results(prob, val, first_node)

    while True:
        
        current_readouts = player.root.N
        while player.root.N < current_readouts + readouts:
            player.tree_search()

        move = player.pick_move()
        player.play_move(move)
        tf.logging.info('playing move: %d hamming distance: %d' % move, state_diff(player.root.position.state))
        if player.root.is_done():
            tf.logging.info('done')
            break



def main(argv):
    network = dual_net.Network()

    play(network)


if __name__ == "__main__":
    tf.app.run()
