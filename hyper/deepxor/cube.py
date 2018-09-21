import os
import itertools
import random
import tensorflow as tf
import numpy as np
import csv
import copy
from queue import Queue
from threading import Thread
import deepxor.dual_net
from deepxor.deepxor import solved, len_solved, num_actions, apply_action, reward, DeepxorModel
from tensorflow.python import keras
from deepxor.hyper_net import Network
from deepxor.selfplay import play
import json

tf.flags.DEFINE_string("model_dir", default=None, help="Estimator model dir")
tf.flags.DEFINE_integer("batch_size", default=8, help="Batch size")
tf.flags.DEFINE_float("learning_rate", default=.1, help="Learning rate")
tf.flags.DEFINE_float("momentum", default=.9, help="momentum")
tf.flags.DEFINE_float("value_weight", default=.1, help="value weight")
tf.flags.DEFINE_integer("train_steps", default=1000, help="training steps")
tf.flags.DEFINE_integer("train_steps_per_eval", default=100, help="training steps per train call")
tf.flags.DEFINE_string("data_file", default="/tmp/predict.tfrecord", help="Input data file")
tf.flags.DEFINE_string("train_file", default="/tmp/train.tfrecord", help="Input data file")
tf.flags.DEFINE_string("job-dir", default="./summary", help="summary output")
tf.flags.DEFINE_integer("rolls", default=150, help="Number of rolls")
tf.flags.DEFINE_integer("rolls_len", default=50, help="Length of one roll")
tf.flags.DEFINE_integer("num_generators", default=2, help="Number of generator threads")
tf.flags.DEFINE_integer("checkpoint_steps", default=500, help="Checkpoint")
tf.flags.DEFINE_integer("eval_steps", default=5, help="When to eval")

FLAGS = tf.flags.FLAGS

def _generate():
    for j in range(0,FLAGS.rolls):
        current = solved
        for i in range(0,FLAGS.rolls_len):
            current = apply_action(current, random.randint(0,num_actions-1))
            for a in range(0, num_actions):
                state = apply_action(current, a)
                yield state, np.zeros([num_actions]), 0, current, reward(state), i + 1

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

def _tensor_map(tensor):
     return {'state': tensor, 'distance': tf.constant([0.])}, {'policy_output': tf.constant(np.zeros([num_actions], dtype='float32')), 'value_output': tf.constant([0.]) , 'parent': tf.constant(np.zeros([len_solved],dtype='float32')), 'reward': tf.constant([0.])}

def predict_input_fn(fname):
    ds = tf.data.TFRecordDataset(fname)
    ds = ds.map(_parse_function)
    return ds.batch(FLAGS.rolls_len*num_actions)

def train_input_fn(fname):
    ds = tf.data.TFRecordDataset(fname)
    ds = ds.map(_parse_function)
    return ds.repeat(count=FLAGS.train_steps_per_eval).shuffle(buffer_size=FLAGS.rolls*FLAGS.rolls_len).batch(FLAGS.batch_size)

def eval_input_fn(tensor):
    ds = tf.data.Dataset.from_tensors(tensor)
    ds = ds.map(_tensor_map)
    return ds.batch(1)

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
        self.input_queue = Queue(maxsize=3)
        self.output_queue = Queue(maxsize=3)
        for f in [ FLAGS.data_file + '-' + str(i) for i in range(0,3)]:
            self.input_queue.put(f)
        self.generator_thread = Thread(target=self.generate_from_queue, daemon=True)
        self.generator_thread.start()

    def generate_from_queue(self):
        while True:
            fname = self.input_queue.get()
            write_samples(fname, _generate)
            self.output_queue.put(fname)

    def get_sample_file(self):
        return self.output_queue.get()

    def put_sample_file(self, fname):
        self.input_queue.put(fname)

def compute_loss(policy_output, value_output, logits, features, labels):
    loss = tf.reduce_mean((FLAGS.value_weight*tf.losses.mean_squared_error(tf.reshape(labels['value_output'], [-1,1]),
        predictions=value_output) + 
        tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels['policy_output'],
            logits=logits)) / features['distance']) + tf.losses.get_regularization_loss()
    return loss

def main(argv):
    tf.logging.set_verbosity(tf.logging.INFO)

    # Create and start a server for the local task.
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.6
    config.gpu_options.allow_growth = True

    generator = AdiGenerator()
    train_samples = []
    tname = FLAGS.train_file
    local_model = DeepxorModel('worker')


    filename_predict = tf.placeholder(tf.string, shape=[])
    filename_training = tf.placeholder(tf.string, shape=[])
    tensor_eval = tf.placeholder(tf.float32, shape=[len_solved])
    trial = json.loads(os.environ.get('TF_CONFIG', '{}')).get('task', {}).get('trial', '')
    eval_path = os.path.join(FLAGS.model_dir, trial, 'hamming_distance')
    hamming_distance = tf.placeholder(tf.int64, shape=[])

    predict_dataset = predict_input_fn(filename_predict)
    training_dataset = train_input_fn(filename_training)
    eval_dataset = eval_input_fn(tensor_eval)

    iterator = tf.data.Iterator.from_structure(predict_dataset.output_types, predict_dataset.output_shapes)
    predict_init_op = iterator.make_initializer(predict_dataset)
    training_init_op = iterator.make_initializer(training_dataset)
    eval_init_op = iterator.make_initializer(eval_dataset)
    next_element = iterator.get_next()

    features, labels = next_element

    policy_output, value_output, logits = local_model(features['state'])

    x_num_actions = FLAGS.rolls_len

    arg = tf.reshape(labels['reward'] + value_output, [x_num_actions, num_actions])
    parent = tf.reshape(labels['parent'], [x_num_actions, num_actions, len_solved])[:,0,:]
    distance = tf.reshape(features['distance'], [x_num_actions, num_actions])[:,0]
    reward = tf.reshape(labels['reward'], [x_num_actions, num_actions])[:,0]
    value = tf.reduce_max(arg, 1)
    policy = tf.one_hot(tf.argmax(arg, 1), num_actions, 1.0, 0.0)

    loss = compute_loss(policy_output, value_output, logits, features, labels)
    global_step = tf.train.get_or_create_global_step()
    learning_rate = tf.train.exponential_decay(FLAGS.learning_rate,
                            global_step, 100000, .96)

    opt = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=FLAGS.momentum)

    train_op = opt.minimize(loss, global_step=global_step)
    hooks=[ tf.train.StopAtStepHook(last_step=FLAGS.train_steps) ]
#            tf.train.LoggingTensorHook(tensors={'step': global_step, 'loss': loss}, every_n_iter=1000)]


    current_step = 0
    next_selfplay = FLAGS.eval_steps

#    summary_hamming_distance = tf.summary.scalar('hamming_distance', hamming_distance)
#    merged_summary_op = tf.summary.merge([summary_hamming_distance])

    output_dir = os.path.join(
        FLAGS.model_dir,
        json.loads(
            os.environ.get('TF_CONFIG', '{}')
        ).get('task', {}).get('trial', '')
    )
    hd = len_solved

    tf.logging.info('Eval path: ' + eval_path)
    summary_writer = tf.summary.FileWriter(eval_path)
    with tf.train.MonitoredTrainingSession(config=config,
                               checkpoint_dir=output_dir,
                               save_checkpoint_secs=None,
                               save_checkpoint_steps=FLAGS.checkpoint_steps,
                               hooks=hooks) as mon_sess:


        network = Network(mon_sess, policy_output, value_output, tensor_eval, eval_init_op)
        while not mon_sess.should_stop():
            tf.logging.info('Loading predict ...')
            fname = generator.get_sample_file()
            if not mon_sess.should_stop():
                mon_sess.run(predict_init_op, feed_dict={filename_predict: fname})

            tf.logging.info('Predicting ...')
            while not mon_sess.should_stop():
                try:
                    _policy, _value, _parent, _reward, _distance = mon_sess.run([policy, value, parent, reward, distance])

                    for a,b,c,d,e,f in zip(_parent, _policy, _value, _parent, _reward, _distance):
                        train_samples.append((a,b,c,d, e, f))
                except tf.errors.OutOfRangeError:
                    break
            generator.put_sample_file(fname)

            tf.logging.info('Writing training data ...')
            write_samples(tname, lambda : train_samples)
            train_samples.clear()

            if not mon_sess.should_stop():
                mon_sess.run(training_init_op, feed_dict={filename_training: tname})
            tf.logging.info('Training ...')
            while not mon_sess.should_stop():
                try:
                    mon_sess.run(train_op)
                except tf.errors.OutOfRangeError:
                    break
            if not mon_sess.should_stop() and next_selfplay == current_step:
                hd = play(network)
                summary = tf.Summary(value=[ tf.Summary.Value(tag="hamming_distance", simple_value=hd), ])
#                summary = mon_sess.run(merged_summary_op, feed_dict={hamming_distance: hd})
                summary_writer.add_summary(summary, global_step)
                summary_writer.flush()
                next_selfplay = current_step + FLAGS.eval_steps
            current_step += 1

        summary_writer.close()

if __name__ == "__main__":
    tf.app.run()


