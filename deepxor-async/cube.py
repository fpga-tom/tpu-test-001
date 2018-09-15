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
from deepxor import solved, len_solved, num_actions, apply_action, reward, DeepxorModel
from tensorflow.python import keras

#tf.enable_eager_execution()

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
tf.flags.DEFINE_integer("num_workers", default=1, help="Number of worker threads")
tf.flags.DEFINE_integer("num_generators", default=2, help="Number of generator threads")
tf.flags.DEFINE_string("ps_hosts", default="localhost:2222", help="Parameter server host")
tf.flags.DEFINE_string("worker_hosts", default="localhost:22232", help="Worker host")
tf.flags.DEFINE_string("job_name", default="ps", help="Id of host")
tf.flags.DEFINE_integer("task_index", default=0, help="Index of host")

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
                yield state, np.zeros([num_actions]), 0, current, reward(state), i
            current = apply_action(current, random.randint(0,num_actions-1))

def _parse_function(example_proto):
     keys_to_features = {'state':tf.FixedLenFeature((len_solved), tf.float32),
                          'policy_output': tf.FixedLenFeature((num_actions), tf.float32),
                          'value_output': tf.FixedLenFeature((1), tf.float32),
                          'parent': tf.FixedLenFeature((len_solved), tf.float32),
                          'reward': tf.FixedLenFeature((1), tf.float32),
                          'distance': tf.FixedLenFeature((1), tf.float32),
                          }
     parsed_features = tf.parse_single_example(example_proto, keys_to_features)
     return {'state': parsed_features['state'], 'distance': parsed_features['distance']}, {'policy_output': parsed_features['policy_output'], 'value_output': parsed_features['value_output'], 'parent': parsed_features['parent'], 'reward': parsed_features['reward']}

def predict_input_fn(fname):
    ds = tf.data.TFRecordDataset(fname)
    ds = ds.map(_parse_function)
    return ds.batch(FLAGS.rolls_len*num_actions)

def train_input_fn(fname):
    ds = tf.data.TFRecordDataset(fname)
    ds = ds.map(_parse_function)
    return ds.repeat().shuffle(buffer_size=FLAGS.rolls*FLAGS.rolls_len).batch(FLAGS.batch_size)

def _floats_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def write_samples(fname, generator):
    writer = tf.python_io.TFRecordWriter(fname)
    for g in generator():
        feature = {'state': _floats_feature(g[0]),
                   'policy_output': _floats_feature(g[1]),
                   'value_output': _floats_feature([g[2]]),
                   'parent': _floats_feature(g[3]),
                   'reward': _floats_feature([g[4]]),
                   'distance': _floats_feature([g[5]])}
        # Create an example protocol buffer
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        # Serialize to string and write on the file
        writer.write(example.SerializeToString())
    writer.close()

    
class AdiGenerator():
    def __init__(self):
        self.input_queue = Queue(maxsize=FLAGS.num_generators)
        self.output_queue = Queue(maxsize=FLAGS.num_generators)
        self.generator_threads = [Thread(target=self.generate_from_queue, daemon=True, args=[x]) for x in range(0, 1)]
        for thread in self.generator_threads:
            thread.start()
        for x in range(0, 1):
            self.input_queue.put(FLAGS.data_file + '-' + str(FLAGS.task_index))

    def generate_from_queue(self, tid):
        while True:
            fname = self.input_queue.get()
            write_samples(fname, _generate)
            self.output_queue.put(fname)

    def get_sample_file(self):
        return self.output_queue.get()

    def put_sample_file(self, fname):
        self.input_queue.put(fname)

class AdiWorker():
    def __init__(self, global_model, opt, generator, wid):
        super(AdiWorker, self).__init__(daemon=True)
        self.wid = wid
        self.opt = opt
        self.global_model = global_model
        self.local_model = DeepxorModel('worker-' + str(self.wid))
        self.generator = generator
        self.fname = FLAGS.data_file + '-' + str(wid)
        self.tname = FLAGS.train_file + '-' + str(wid)
        self.save_file = FLAGS.model_dir
        self.train_samples = []
        self.predict = True

    def run(self):
        filename = tf.placeholder(tf.string, shape=[])
        predict_dataset = predict_input_fn(filename)
        training_dataset = train_input_fn(filename)
        iterator = tf.data.Iterator.from_structure(predict_dataset.output_types, predict_dataset.output_shapes)
        next_element = iterator.get_next()
        predict_init_op = iterator.make_initializer(predict_dataset)
        training_init_op = iterator.make_initializer(training_dataset)

        features, labels = next_element

        policy_output, value_output, logits = self.local_model(features['state'])

        x_num_actions = FLAGS.rolls_len

        arg = tf.reshape(labels['reward'] + value_output, [x_num_actions, num_actions])
        parent = tf.reshape(labels['parent'], [x_num_actions, num_actions, len_solved])[:,0,:]
        distance = tf.reshape(features['distance'], [x_num_actions, num_actions])[:,0]
        reward = tf.reduce_max(arg, 1)
        policy = tf.one_hot(tf.argmax(arg, 1), num_actions, 1.0, 0.0)

        with tf.GradientTape(persistent=True) as tape:
            total_loss = self.compute_loss(policy_output, value_output, logits, features, labels)
        grads = self.opt.compute_gradients(total_loss, self.local_model.trainable_weights)
        print([g for g in zip(grads, self.global_model.trainable_weights)])
        update_op = self.opt.apply_gradients(zip(grads, self.global_model.trainable_weights))

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            while True:
                fname = self.generator.get_sample_file()
                sess.run(predict_init_op, feed_dict={filename: fname})
                _policy, _value, _parent, _reward, _distance = sess.run([policy_output, value_output, parent, reward, distance])
                self.generator.put_sample_file(fname)

                for a,b,c,d,e,f in zip(_parent, _policy, _value, _parent, _reward, _distance):
                    self.train_samples.append((a,b,c,d, e, f))

                write_samples(self.tname, lambda : [s for s in self.train_samples])
                self.train_samples.clear()

                sess.run(training_init_op, feed_dict={filename: self.tname})
                sess.run(update_op)

                # Update local model with new weights
                self.local_model.set_weights(self.global_model.get_weights())
                print(total_loss)
        del tape

    def compute_loss(self, policy_output, value_output, logits, features, labels):
        loss = tf.reduce_mean((0.5*tf.losses.mean_squared_error(tf.reshape(labels['value_output'], [-1,1]),
            predictions=value_output) + 
            tf.nn.softmax_cross_entropy_with_logits(labels=labels['policy_output'],
                logits=logits)) / (features['distance'] + 1.0))
        return loss


class AdiMaster():
    def __init__(self):

        self.opt = tf.train.GradientDescentOptimizer(learning_rate=FLAGS.learning_rate, use_locking=True)
        self.global_model = DeepxorModel('master')
        policy_output, value_output, logits = self.global_model(tf.convert_to_tensor(np.random.random((1, len_solved)), dtype=tf.float32))
        self.generator = AdiGenerator()
        self.workers = [AdiWorker(self.global_model, self.opt, self.generator, x) for x in range(0, FLAGS.num_workers)]

    def train(self):
        for i,w in enumerate(self.workers):
            tf.logging.info("Starting worker {}".format(i))
            w.start()

        [w.join() for w in self.workers]



def adi(est, cpu_est):
    workers = [AdiWorker('worker', x) for x in range(0, FLAGS.num_workers)]
    worker_threads = [Thread(target=workers[i].run, daemon=True) for x in range(0,FLAGS.num_workers)]
    for thread in worker_threads:
        thread.start()
        

    current_step = estimator._load_global_step_from_checkpoint_dir(FLAGS.model_dir)
    while current_step < FLAGS.train_steps:
        next_checkpoint = min(current_step + FLAGS.train_steps_per_eval,
                          FLAGS.train_steps)
        tf.logging.info("Type %s" % type(next_checkpoint))

        tf.logging.info('Generating ' + g_fname)
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

        generator.put_sample_file(g_fname)
        tf.logging.info('Training ...')
        est.train(train_input_fn, max_steps=next_checkpoint)
        current_step = next_checkpoint


def model_fn(features, labels, mode, params):
    with tf.variable_scope(params['scope']):
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

def compute_loss(policy_output, value_output, logits, features, labels):
    loss = tf.reduce_mean((0.5*tf.losses.mean_squared_error(tf.reshape(labels['value_output'], [-1,1]),
        predictions=value_output) + 
        tf.nn.softmax_cross_entropy_with_logits(labels=labels['policy_output'],
            logits=logits)) / (features['distance'] + 1.0))
    return loss

def main(argv):
    tf.logging.set_verbosity(tf.logging.INFO)
    ps_hosts = FLAGS.ps_hosts.split(",")
    worker_hosts = FLAGS.worker_hosts.split(",")

    # Create a cluster from the parameter server and worker hosts.
    cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})

    # Create and start a server for the local task.
    server = tf.train.Server(cluster,
                           job_name=FLAGS.job_name,
                           task_index=FLAGS.task_index)

    if FLAGS.job_name == "ps":
        server.join()
    elif FLAGS.job_name == "worker":
        generator = AdiGenerator()
        train_samples = []
        tname = FLAGS.train_file + '-' + str(FLAGS.task_index)
        local_model = DeepxorModel('worker-' + str(FLAGS.task_index))
        with tf.device(tf.train.replica_device_setter(
            worker_device="/job:worker/task:%d" % FLAGS.task_index,
            cluster=cluster)):

                filename = tf.placeholder(tf.string, shape=[])
                predict_dataset = predict_input_fn(filename)
                training_dataset = train_input_fn(filename)
                iterator = tf.data.Iterator.from_structure(predict_dataset.output_types, predict_dataset.output_shapes)
                next_element = iterator.get_next()
                predict_init_op = iterator.make_initializer(predict_dataset)
                training_init_op = iterator.make_initializer(training_dataset)

                features, labels = next_element

                policy_output, value_output, logits = local_model(features['state'])

                x_num_actions = FLAGS.rolls_len

                arg = tf.reshape(labels['reward'] + value_output, [x_num_actions, num_actions])
                parent = tf.reshape(labels['parent'], [x_num_actions, num_actions, len_solved])[:,0,:]
                distance = tf.reshape(features['distance'], [x_num_actions, num_actions])[:,0]
                reward = tf.reduce_max(arg, 1)
                policy = tf.one_hot(tf.argmax(arg, 1), num_actions, 1.0, 0.0)

                loss = compute_loss(policy_output, value_output, logits, features, labels)
                global_step = tf.contrib.framework.get_or_create_global_step()
                train_op = tf.train.GradientDescentOptimizer(learning_rate=FLAGS.learning_rate).minimize(loss, global_step=global_step)
                hooks=[tf.train.StopAtStepHook(last_step=1000000)]

                with tf.train.MonitoredTrainingSession(master=server.target,
                                           is_chief=(FLAGS.task_index == 0),
                                           checkpoint_dir="/tmp/train_logs" + str(FLAGS.task_index),
                                           hooks=hooks) as mon_sess:
                    while not mon_sess.should_stop():
                        fname = generator.get_sample_file()
                        mon_sess.run(predict_init_op, feed_dict={filename: fname})
                        _policy, _value, _parent, _reward, _distance = mon_sess.run([policy_output, value_output, parent, reward, distance])
                        generator.put_sample_file(fname)

                        for a,b,c,d,e,f in zip(_parent, _policy, _value, _parent, _reward, _distance):
                            train_samples.append((a,b,c,d, e, f))

                        write_samples(tname, lambda : [s for s in train_samples])
                        train_samples.clear()

                        mon_sess.run(training_init_op, feed_dict={filename: tname})
                        mon_sess.run(train_op)




if __name__ == "__main__":
    tf.app.run()


