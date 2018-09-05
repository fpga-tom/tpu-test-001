import random
import tensorflow as tf
import numpy as np

tf.flags.DEFINE_string("tpu", default=None, help="TPU which to use")
tf.flags.DEFINE_string("tpu_zone", default=None, help="GCE zone of TPU" )
tf.flags.DEFINE_string("gcp_project", default=None, help="Project name of TPU enabled project")

tf.flags.DEFINE_bool("use_tpu", default=False, help="Use TPU rather than CPU")
tf.flags.DEFINE_string("model_dir", default=None, help="Estimator model dir")
tf.flags.DEFINE_integer("batch_size", default=8, help="Batch size")
tf.flags.DEFINE_integer("iterations", default=50, help="Number of iterations per TPU loop")
tf.flags.DEFINE_integer("num_shards", default=8, help="Number of shards (TPU chips)")
tf.flags.DEFINE_float("learning_rate", default=.1, help="Learning rate")
tf.flags.DEFINE_integer("train_steps", default=1000, help="training steps")
tf.flags.DEFINE_string("data_file", default="./data.csv", help="Input data file")

FLAGS = tf.flags.FLAGS

solved = [1, 1, 1, 1, 0, 0, 0, 0]
num_actions = len(solved)*2

inference_output = None
inferene_input = (tf.placeholder(tf.float32, [None, len(solved)], name='inference_input'))

def apply_action(state, action):
    state = [i for i in state]
    state[action >> 1] ^= action & 1
    return state

def reward(state):
    if all([_solved == _state for _solved, _state in zip(solved, state)]):
        return 1
    return -1

def generate(k):
    result = [(solved, 0)]
    current = solved
    for i in range(0,k):
        state = apply_action(current, random.randint(0,num_actions-1))
        result.append((state, i+1))
        current = state
    return result

def generate_all(l, k):
    result = []
    for i in range(0, l):
        g = generate(k)
        for j in g:
            result.append(j)
    return result

def adi():
    X = generate_all(100,8)
    for x in X:
        for a in range(0, num_actions):
            outputs = sess.run(inference_output, feed_dict={inference_input: apply_action(x, a)})


def model_fn(features, labels, mode, params):
    l_0 = tf.layers.dense(features, 4096, activation=tf.nn.elu)
    l_1 = tf.layers.dense(l_0, 2048, activation=tf.nn.elu)
    l_2 = tf.layers.dense(l_1, 512, activation=tf.nn.elu)
    l_3 = tf.layers.dense(l_1, 512, activation=tf.nn.elu)
    policy_output = tf.nn.softmax(tf.layers.dense(l_2, num_actions, activation=tf.nn.sigmoid))
    value_output = tf.layers.dense(l_3, 1, activation=tf.tanh)

    loss = tf.reduce_mean(1e-3*tf.losses.mean_squared_error(labels['value_output'], predictions=value_output) +
            tf.losses.softmax_cross_entropy_with_logits_v2(labels=labels['policy_output'], logits=policy_output))

    if mode == tf.estimator.ModeKeys.TRAIN:
        learning_rate = tf.train.exponential_decay(FLAGS.learning_rate,
                tf.train.get_global_step(), 100000, .96)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        if FLAGS.use_tpu:
            optimizer = tf.contrib.tpu.CrossShardOptimizer(optimizer)

    predictions = {
            'policy_output' : policy_output,
            'value_output' : value_output
    }

    tpu_estimator_spec = tf.contrib.tpu.TPUEstimatorSpec(
            mode=mode,
            predictions=predictions,
            loss=loss, 
            train_op=optimizer.minimize(loss, tf.train.get_global_step())
    )

    if FLAGS.use_tpu:
        return tpu_estimator_spec
    else:
        return tpu_estimator_spec.as_estimator_spec()
    


def main(argv):
    tf.logging.set_verbosity(tf.logging.INFO)

    if FLAGS.use_tpu:
        tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
            FLAGS.tpu,
            zone=FLAGS.tpu_zone,
            project=FLAGS.gcp_project
        )
    else:
        tpu_cluster_resolver = ''

    run_config = tf.contrib.tpu.RunConfig(
        cluster=tpu_cluster_resolver,
        model_dir=FLAGS.model_dir,
        session_config=tf.ConfigProto(
            allow_soft_placement=True, log_device_placement=True
            ),
        tpu_config=tf.contrib.tpu.TPUConfig(FLAGS.iterations, FLAGS.num_shards)
    )

    estimator = tf.contrib.tpu.TPUEstimator(
        model_fn=model_fn,
        train_batch_size=FLAGS.batch_size,
        use_tpu=FLAGS.use_tpu,
        params={'data_file': FLAGS.data_file},
        config=run_config
    )

    inference_output = estimator.predictions

    with tf.Session as sess:
        adi(sess)


if __name__ == "__main__":
    tf.app.run()

print(gen_n(10,5))

