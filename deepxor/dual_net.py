import tensorflow as tf
from deepxor import num_actions, len_solved
from queue import Queue
from threading import Thread


FLAGS = tf.flags.FLAGS

def create(input_layer, num_actions):
    l_0 = tf.layers.dense(input_layer, 4096, activation=tf.nn.elu)
    l_1 = tf.layers.dense(l_0, 2048, activation=tf.nn.elu)
    l_2 = tf.layers.dense(l_1, 512, activation=tf.nn.elu)
    l_3 = tf.layers.dense(l_1, 512, activation=tf.nn.elu)
    logits = tf.layers.dense(l_2, num_actions)
    policy_output = tf.nn.softmax(logits)
    value_output = tf.layers.dense(l_3, 1, activation=tf.tanh)

    return policy_output, value_output, logits

def model_fn(features, labels, mode, params):
    policy_output, value_output, logits = create(features, num_actions)

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
                'policy_output' : policy_output,
                'value_output' : value_output,
        }
        tpu_estimator_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                predictions=predictions
        )

    if FLAGS.use_tpu:
        return tpu_estimator_spec
    else:
        return tpu_estimator_spec.as_estimator_spec()


class Network():

    def __init__(self):
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

        self.estimator = tf.contrib.tpu.TPUEstimator(
            model_fn=model_fn,
            train_batch_size=FLAGS.batch_size,
            predict_batch_size=FLAGS.batch_size,
            use_tpu=FLAGS.use_tpu,
            config=run_config
        )

        self.input_queue = Queue(maxsize=1)
        self.output_queue = Queue(maxsize=1)

        self.prediction_thread = Thread(target=self.predict_from_queue, daemon=True)
        self.prediction_thread.start()

    def generate_from_queue(self):
        while True:
            yield self.input_queue.get()

    def predict_from_queue(self):

        for i in self.estimator.predict(input_fn=self.queued_predict_input_fn):
            self.output_queue.put(i)

    def predict(self, features):
        self.input_queue.put(features)
        predictions = self.output_queue.get()
        return predictions['policy_output'], predictions['value_output']

    def queued_predict_input_fn(self, params):
        dataset = tf.data.Dataset.from_generator(self.generate_from_queue,
                (tf.float32), (tf.TensorShape([len_solved])))
        dataset = dataset.map(lambda x : tf.reshape(x, [1, len_solved]))
        return dataset


    def run_many(self, positions):
        return [self.predict(p.state) for p in positions]
    
