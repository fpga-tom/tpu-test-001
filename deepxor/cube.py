import itertools
import random
import tensorflow as tf
import numpy as np
import csv

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
tf.flags.DEFINE_string("data_file", default="./x_input.csv", help="Input data file")
tf.flags.DEFINE_string("train_file", default="./train.csv", help="Input data file")
tf.flags.DEFINE_string("sample_file", default="./X_input.csv", help="Samples data file")

FLAGS = tf.flags.FLAGS

solved = [1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1]
len_solved = len(solved)
num_actions = len_solved + 1

FIELD_DEFAULTS=[[0.] for i in range(0, len_solved)] + [[0.], [0.]]
FIELD_TRAIN=[[0.] for i in range(0, len_solved)] + [[0.]] + [[0.] for i in range(0, num_actions)] + [[0.]]
COLUMNS = ['state'] + ['parent'] + ['reward'] + ['distance']
COLUMNS_TRAIN = ['state', 'policy_output', 'value_output', 'distance']
feature_columns = [tf.feature_column. numeric_column(name, shape=(len_solved)) for name in COLUMNS[:-3]]
feature_columns_train = [tf.feature_column.numeric_column(name) for name in COLUMNS[:-3]]

def apply_action(state, action):
    state = [i for i in state]
    if action < len(solved):
        state[action] ^= 1
    return state

def reward(state):
    if all([_solved == _state for _solved, _state in zip(solved, state)]):
        return 1
    return -1

def _generate():
    for j in range(0,150):
        current = solved
        for i in range(0,30):
            for a in range(0, num_actions):
                state = apply_action(current, a)
                yield state, current, reward(state), i
            current = apply_action(current, random.randint(0,num_actions-1))

def predict_input_fn(params):
    ds = tf.data.Dataset.from_generator(_generate,
            (tf.float32, tf.float32, tf.float32, tf.float32),
            (tf.TensorShape([len_solved]), tf.TensorShape([len_solved]),  tf.TensorShape([]), tf.TensorShape([])))
    ds = ds.map(lambda s, c, r, i: ({'state': s, 'parent': c, 'reward': r, 'distance': i},{}))
    return ds.batch(FLAGS.batch_size)

train_samples = []

def train_generate():
    for sample in train_samples:
        yield sample

def train_input_fn(params):
    ds = tf.data.Dataset.from_generator(train_generate,
            (tf.float32, tf.float32, tf.float32, tf.float32),
            (tf.TensorShape([len_solved]), tf.TensorShape([len_solved+1]),  tf.TensorShape([]), tf.TensorShape([])))
    ds = ds.map(lambda s, c, r, i: ({'state': s}, {'policy_output': c, 'value_output': r, 'distance': i}))
    return ds.repeat().shuffle(buffer_size=50000).apply(tf.contrib.data.batch_and_drop_remainder(FLAGS.batch_size)).make_one_shot_iterator().get_next()


def adi(estimator):
    global train_samples
    for c in range(0,100):
        train_samples = []
        outputs = estimator.predict(predict_input_fn)
        buf = []
        for o in outputs:
            buf.append(o)
            if len(buf) == num_actions:
                arg = [x['reward'][0] for x in buf]
                y_v = np.max(arg)
                y_p = [0 for i in range(0, num_actions)]
                y_p[np.argmax(arg)] = 1
                train_samples.append((buf[0]['parent'], y_p, y_v, buf[0]['distance']))
                buf = []

        estimator.train(train_input_fn, max_steps=FLAGS.train_steps*(c+1))


def model_fn(features, labels, mode, params):
    input_layer = tf.feature_column.input_layer(features, feature_columns)
    l_0 = tf.layers.dense(input_layer, 4096, activation=tf.nn.elu)
    l_1 = tf.layers.dense(l_0, 2048, activation=tf.nn.elu)
    l_2 = tf.layers.dense(l_1, 512, activation=tf.nn.elu)
    l_3 = tf.layers.dense(l_1, 512, activation=tf.nn.elu)
    logits = tf.layers.dense(l_2, num_actions)
    policy_output = tf.nn.softmax(logits)
    value_output = tf.layers.dense(l_3, 1, activation=tf.tanh)


    if mode == tf.estimator.ModeKeys.TRAIN:
        loss = tf.reduce_mean((1e-2*tf.losses.mean_squared_error(tf.reshape(labels['value_output'],[-1,1]),
            predictions=value_output) + 
            tf.nn.softmax_cross_entropy_with_logits(labels=labels['policy_output'],
                logits=logits)) / (labels['distance'] + 1))
        learning_rate = tf.train.exponential_decay(FLAGS.learning_rate,
                tf.train.get_global_step(), 100000, .96)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        if FLAGS.use_tpu:
            optimizer = tf.contrib.tpu.CrossShardOptimizer(optimizer)

        tpu_estimator_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=loss, 
                train_op=optimizer.minimize(loss, tf.train.get_global_step())
        )
        if FLAGS.use_tpu:
            return tpu_estimator_spec
        else:
            return tpu_estimator_spec.as_estimator_spec()

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
                'policy_output' : policy_output,
                'value_output' : value_output,
#                'reward': tf.reshape(features['reward'],[-1,1]) + value_output,
#                'parent': features['parent'],
#                'distance': features['distance']
        }
        tpu_estimator_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                predictions=predictions
        )
        return tpu_estimator_spec


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
        predict_batch_size=FLAGS.batch_size,
        use_tpu=FLAGS.use_tpu,
        params={'data_file': FLAGS.data_file, 'train_file': FLAGS.train_file},
        config=run_config
    )


    adi(estimator)


if __name__ == "__main__":
    tf.app.run()

print(gen_n(10,5))

