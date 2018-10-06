class Network():

    def __init__(self, sess, policy_output, value_output, tensor_eval, eval_init_op):
        self.sess = sess
        self.policy_output = policy_output
        self.value_output = value_output
        self.tensor_eval = tensor_eval
        self.eval_init_op = eval_init_op

    def predict(self, features):
        self.sess.run(self.eval_init_op, feed_dict={self.tensor_eval: features})
        policy, value = self.sess.run([self.policy_output, self.value_output])
        return policy[0], value[0]

    def run_many(self, positions):
        return zip(*[self.predict(p.state) for p in positions])
