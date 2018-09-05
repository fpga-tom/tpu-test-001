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

FLAGS = tf.flags.FLAGS

len_solved = 8
num_actions = len_solved + 1

FIELD_DEFAULTS=[[0.] for i in range(0, len_solved)] + [[0.], [0.]]
COLUMNS = ['a'+str(i) for i in range(0, len_solved)] + ['reward'] + ['distance']
feature_columns = [tf.feature_column.numeric_column(name) for name in COLUMNS[:-2]]

def _parse_line(line):
    fields = tf.decode_csv(line, FIELD_DEFAULTS)
    features = dict(zip(COLUMNS, fields))
    return features

def predict_input_fn(params):
    data_file=params['data_file']
    ds = tf.data.TextLineDataset(data_file)
    ds = ds.map(_parse_line)
    return ds.apply(tf.contrib.data.batch_and_drop_remainder(FLAGS.batch_size)).make_one_shot_iterator().get_next()


def adi(estimator):
    outputs = estimator.predict(predict_input_fn)
    buf = []
    with open('./train.csv', 'wb') as csvfile:
        writer = csv.writer(csvfile)
        for i,o in enumerate(outputs):
            buf.append(o)
            if i % num_actions == num_actions-1:
                arg = [x['reward'][0] for x in buf]
                y_v = np.max(arg)
                y_p = np.argmax(arg)
                writer.writerow(o['policy_output'] + o['value_output'])
                buf = []


def model_fn(features, labels, mode, params):
    input_layer = tf.feature_column.input_layer(features, feature_columns)
    l_0 = tf.layers.dense(input_layer, 4096, activation=tf.nn.elu)
    l_1 = tf.layers.dense(l_0, 2048, activation=tf.nn.elu)
    l_2 = tf.layers.dense(l_1, 512, activation=tf.nn.elu)
    l_3 = tf.layers.dense(l_1, 512, activation=tf.nn.elu)
    policy_output = tf.nn.softmax(tf.layers.dense(l_2, num_actions, activation=tf.nn.sigmoid))
    value_output = tf.layers.dense(l_3, 1, activation=tf.tanh)


    if mode == tf.estimator.ModeKeys.TRAIN:
        loss = tf.reduce_mean(1e-3*tf.losses.mean_squared_error(labels['value_output'],
            predictions=value_output) + 
            tf.losses.softmax_cross_entropy_with_logits_v2(labels=labels['policy_output'],
                logits=policy_output))
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

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
                'policy_output' : policy_output,
                'value_output' : value_output,
                'reward': tf.reshape(features['reward'],[-1,1]) + value_output
        }
        tpu_estimator_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                predictions=predictions
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

    adi(estimator)


if __name__ == "__main__":
    tf.app.run()

print(gen_n(10,5))

