import itertools
import random
import tensorflow as tf
import numpy as np
import csv
import copy
from queue import Queue
from threading import Thread
from tensorflow.python.estimator import estimator
import dual_net
from deepxor import solved, len_solved, num_actions, apply_action, reward


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
tf.flags.DEFINE_integer("train_steps_per_eval", default=100, help="training steps per train call")
tf.flags.DEFINE_string("data_file", default="./predict.tfrecord", help="Input data file")
tf.flags.DEFINE_string("train_file", default="./train.tfrecord", help="Input data file")
tf.flags.DEFINE_string("sample_file", default="./X_input.tfrecord", help="Samples data file")
tf.flags.DEFINE_integer("rolls", default=150, help="Number of rolls")
tf.flags.DEFINE_integer("rolls_len", default=50, help="Length of one roll")
tf.flags.DEFINE_integer("num_workers", default=4, help="Number of worker threads")

FLAGS = tf.flags.FLAGS


FIELD_DEFAULTS=[[0.] for i in range(0, len_solved)] + [[0.], [0.]]
FIELD_TRAIN=[[0.] for i in range(0, len_solved)] + [[0.]] + [[0.] for i in range(0, num_actions)] + [[0.]]
COLUMNS = ['state'] + ['parent'] + ['reward'] + ['distance']
COLUMNS_TRAIN = ['state', 'policy_output', 'value_output', 'distance']
feature_columns = [tf.feature_column. numeric_column(name, shape=(len_solved)) for name in COLUMNS[:-3]]
feature_columns_train = [tf.feature_column.numeric_column(name) for name in COLUMNS[:-3]]


def _generate():
    for j in range(0,FLAGS.rolls):
        current = solved
        for i in range(0,FLAGS.rolls_len):
            for a in range(0, num_actions):
                state = apply_action(current, a)
                yield state, current, reward(state), i
            current = apply_action(current, random.randint(0,num_actions-1))

def _parse_function(example_proto):
     keys_to_features = {'state':tf.FixedLenFeature((len_solved), tf.float32),
                          'parent': tf.FixedLenFeature((len_solved), tf.float32),
                          'reward': tf.FixedLenFeature((1), tf.float32),
                          'distance': tf.FixedLenFeature((1), tf.float32),
                          }
     parsed_features = tf.parse_single_example(example_proto, keys_to_features)
     return parsed_features

def predict_input_fn(params):
    ds = tf.data.TFRecordDataset(FLAGS.data_file)
    ds = ds.map(_parse_function)
#    ds = ds.map(lambda s, c, r, i: {'state': s, 'parent': c, 'reward': r, 'distance': i})
    return ds.batch(2**14)

def train_generate():
    for sample in train_samples:
        yield sample

def _parse_train(example_proto):
     keys_to_features = {'state':tf.FixedLenFeature((len_solved), tf.float32),
                          'policy_output': tf.FixedLenFeature((num_actions), tf.float32),
                          'value_output': tf.FixedLenFeature((1), tf.float32),
                          'distance': tf.FixedLenFeature((1), tf.float32),
                          }
     parsed_features = tf.parse_single_example(example_proto, keys_to_features)
     return {'state': parsed_features['state']}, {'policy_output': parsed_features['policy_output'], 'value_output': parsed_features['value_output'], 'distance': parsed_features['distance']}

def train_input_fn(params):
    ds = tf.data.TFRecordDataset(FLAGS.train_file)
    ds = ds.map(_parse_train)
#    ds = ds.map(lambda s, c, r, i: ({'state': s}, {'policy_output': c, 'value_output': r, 'distance': i}))
    return ds.repeat().shuffle(buffer_size=50000).apply(tf.contrib.data.batch_and_drop_remainder(FLAGS.batch_size))

def _floats_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def generate_samples():
    writer = tf.python_io.TFRecordWriter(FLAGS.data_file)
    for g in _generate():
        feature = {'state': _floats_feature(g[0]),
                   'parent': _floats_feature(g[1]),
                   'reward': _floats_feature([g[2]]),
                   'distance': _floats_feature([g[3]])}
        # Create an example protocol buffer
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        # Serialize to string and write on the file
        writer.write(example.SerializeToString())
    writer.close()

def write_train_samples(train_samples):
    writer = tf.python_io.TFRecordWriter(FLAGS.train_file)
    for g in train_samples:
        feature = {'state': _floats_feature(g[0]),
                   'policy_output': _floats_feature(g[1]),
                   'value_output': _floats_feature([g[2]]),
                   'distance': _floats_feature([g[3]])}
        # Create an example protocol buffer
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        # Serialize to string and write on the file
        writer.write(example.SerializeToString())
    writer.close()

def adi(est, cpu_est):
    class AdiWorker():
        def __init__(self):
            self.input_queue = Queue(maxsize=FLAGS.num_workers)
            self.output_queue = Queue(maxsize=FLAGS.num_workers)
            self.train_samples = []
            self.worker_threads = [Thread(target=self.work_from_queue, daemon=True) for x in range(0,FLAGS.num_workers)]
            for thread in self.worker_threads:
                thread.start()
            
            self.writer_thread = Thread(target=self.write_from_queue, daemon=True)
            self.writer_thread.start()

        def work_from_queue(self):
            while True:
                buf = self.input_queue.get()
                if len(buf) == num_actions:
                    arg = [x['reward'][0] for x in buf]
                    y_v = np.max(arg)
                    y_p = [0 for i in range(0, num_actions)]
                    y_p[np.argmax(arg)] = 1
                    self.output_queue.put((buf[0]['parent'], y_p, y_v, buf[0]['distance']))

        def write_from_queue(self):
            while True:
                self.train_samples.append(self.output_queue.get())

        def write(self):
            write_train_samples(self.train_samples)
            self.train_samples.clear()
        
        def process(self, buf):
            self.input_queue.put(buf)


    worker = AdiWorker()
    current_step = estimator._load_global_step_from_checkpoint_dir(FLAGS.model_dir)
    while current_step < FLAGS.train_steps:
        next_checkpoint = min(current_step + FLAGS.train_steps_per_eval,
                          FLAGS.train_steps)
        tf.logging.info("Type %s" % type(next_checkpoint))

        tf.logging.info('Generating ...')
        generate_samples()
        tf.logging.info('Predicting ...')
        outputs = cpu_est.predict(predict_input_fn)
        buf = []
        for o in outputs:
            buf.append(o)
            if len(buf) == num_actions:
                worker.process(buf)
                buf = []

        tf.logging.info('Writing ...')
        worker.write()

        tf.logging.info('Training ...')
        est.train(train_input_fn, max_steps=next_checkpoint)
        current_step = next_checkpoint


def model_fn(features, labels, mode, params):
    input_layer = tf.feature_column.input_layer(features, feature_columns)
    policy_output, value_output, logits = dual_net.create(input_layer, num_actions)

    if mode == tf.estimator.ModeKeys.TRAIN:
        loss = tf.reduce_mean((tf.losses.mean_squared_error(tf.reshape(labels['value_output'],[-1,1]),
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

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
                'policy_output' : policy_output,
                'value_output' : value_output,
                'reward': tf.reshape(features['reward'],[-1,1]) + value_output,
                'parent': features['parent'],
                'distance': features['distance']
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
    with tf.device('/gpu:0'):
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
                allow_soft_placement=True, log_device_placement=False
                ),
            tpu_config=tf.contrib.tpu.TPUConfig(FLAGS.iterations, FLAGS.num_shards)
        )

        est= tf.contrib.tpu.TPUEstimator(
            model_fn=model_fn,
            train_batch_size=FLAGS.batch_size,
            predict_batch_size=FLAGS.batch_size,
            use_tpu=FLAGS.use_tpu,
            params={'data_file': FLAGS.data_file, 'train_file': FLAGS.train_file},
            config=run_config
        )

        cpu_est= tf.contrib.tpu.TPUEstimator(
            model_fn=model_fn,
            train_batch_size=FLAGS.batch_size,
            predict_batch_size=2**14,
            use_tpu=False,
            params={},
            config=run_config
        )

        adi(est, cpu_est)


if __name__ == "__main__":
    tf.app.run()


