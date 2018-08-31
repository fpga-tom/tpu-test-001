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


COLUMNS=['a','b','label']
feature_columns = [tf.feature_column.numeric_column(name) for name in COLUMNS[:-1]]

def model_fn(features, labels, mode, params):
    del params

    input_layer = tf.feature_column.input_layer(features, feature_columns)
    dense0 = tf.layers.dense(input_layer, 8, activation=tf.tanh, name="dense0")
    dense1 = tf.layers.dense(dense0, 1, activation=tf.sigmoid, name="dense1")

    loss = tf.reduce_mean(tf.losses.mean_squared_error(labels=labels, predictions=dense1))

    if mode == tf.estimator.ModeKeys.TRAIN:
        learning_rate = tf.train.exponential_decay(FLAGS.learning_rate,
                tf.train.get_global_step(), 100000, .96)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        if FLAGS.use_tpu:
            optimizer = tf.contrib.tpu.CrossShardOptimizer(optimizer)
        return tf.contrib.tpu.TPUEstimatorSpec(mode=mode,loss=loss,
                train_op=optimizer.minimize(loss, tf.train.get_global_step())
            )

def _parse_line(line):
    FIELD_DEFAULTS=[[0.],[0.],[0.]]
    fields = tf.decode_csv(line, FIELD_DEFAULTS)
    features = dict(zip(COLUMNS, fields))
    label = features.pop('label')
    return features, tf.reshape(label, [1])

def train_input_fn(params):
    data_file=params['data_file']
    ds = tf.data.TextLineDataset(data_file)
    ds = ds.map(_parse_line)
    return ds.batch(FLAGS.batch_size).repeat().make_one_shot_iterator().get_next()


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

    estimator.train(input_fn=train_input_fn, max_steps=FLAGS.train_steps)


if __name__ == "__main__":
    tf.app.run()
