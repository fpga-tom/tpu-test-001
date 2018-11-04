import tensorflow as tf
import numpy as np

unwrap = 5
variance = .7
codeword_len = 127
batch_size=40
msgTx=[1,1,0,1,1,0,0,1,1,1,0,1,1,0,1,0,0,1,1,1,1,0,1,1,1,1,1,0,1,0,1,0,0,0,0,1,1,0,1,0,0,0,1,1,0]
msgEnc=[1,1,0,1,1,0,0,1,1,1,0,1,1,0,1,0,0,1,1,1,1,0,1,1,1,1,1,0,1,0,1,0,0,0,0,1,1,0,1,0,0,0,1,1,0,1,0,0,0,0,1,1,0,0,0,1,1,0,0,0,0,0,0]
msgTx=[1,1,0,0,0,1,0,1,1,1,0,0,0,1,0,1,1,1,0,0,0,1,0,1,0,0,0,0,1,1,1,0,1,1,0,1,0,0,0,0,0,0,0,0,1,1,0,0,0,1,0,0,1,0,0,0,0,0,1,1,1,0,0,0]
msgEnc=[1,1,0,0,0,1,0,1,1,1,0,0,0,1,0,1,1,1,0,0,0,1,0,1,0,0,0,0,1,1,1,0,1,1,0,1,0,0,0,0,0,0,0,0,1,1,0,0,0,1,0,0,1,0,0,0,0,0,1,1,1,0,0,0,0,0,1,1,0,1,0,0,0,0,0,0,0,0,1,0,0,0,1,0,1,0,1,0,0,1,1,1,0,0,1,1,0,1,0,0,0,0,1,0,1,0,0,0,0,0,0,0,1,0,0,0,1,1,1,0,0,0,0,0,0,1,1]


tf.flags.DEFINE_integer("checkpoint_steps", default=5000, help="Checkpoint")
tf.flags.DEFINE_string("model_dir", default=None, help="Estimator model dir")
tf.flags.DEFINE_integer("train_steps", default=100000, help="training steps")

FLAGS = tf.flags.FLAGS

def create_model(unwrap_depth):
    with open('bch_127_64.txt', 'r') as f:
        lines = [line.split(',') for line in  f.read().splitlines()]
        lines = [(int(l[0]), int(l[1])) for l in lines]
        num_edges_tanner = len(lines)
        rows = {}
        cols = {}
        line = {}
        line_rev = {}
        for lineno, l in enumerate(lines):
            if not rows.has_key(l[0]):
                rows[l[0]] = []
            rows[l[0]] += [l[1]]
            if not cols.has_key(l[1]):
                cols[l[1]] = []
            cols[l[1]] += [l[0]]
            line[l[0],l[1]] = lineno
            line_rev[lineno] = (l[0], l[1])

        lv = tf.placeholder(tf.float32, shape=(None, unwrap_depth,codeword_len))
        init_input = tf.placeholder(tf.float32, shape=(None, num_edges_tanner))

        indices = {}
        indices1 = {}
        indices2 = {}

        for v, cl in cols.iteritems():
            for c in cl:
                indices[c, v] = len(indices)
                for cc in cl:
                    if cc != c:
                        indices1[c,v,cc] = len(indices1)
                for vv in rows[c]:
                    if vv == v:
                        indices2[c,v,vv] = len(indices2)


        print(len(indices))
        print(len(indices1))
        print(len(indices2))
        print(len(line))


        lv_series = tf.unstack(lv, unwrap_depth, axis=1)
        filter_matrix = np.zeros([num_edges_tanner, num_edges_tanner])
        filter_matrix1 = np.zeros([codeword_len, num_edges_tanner])
        sp_indices1 = []
        sp_indices2 = []
        for e in lines:
            filter_matrix1[e[1]-1,line[e]] = 1.
            sp_indices2 += [[e[1]-1, line[e]]]
            for c in cols[e[1]]:
                _e = (c, e[1])
                if c != e[0]:
                    filter_matrix[line[e],line[_e]] = 1.
                    sp_indices1 += [[line[e], line[_e]]]

        count = 0
        next_input = init_input
        abc = tf.Variable(np.random.rand(len(sp_indices1)), name='var2', dtype=tf.float32)
        wv = tf.Variable(np.random.rand(codeword_len), name='var1',dtype=tf.float32)


        codeword = [e[1]-1 for e in lines]
        codeword0 = [e[1]-1 for e in lines]
        codeword0 = set(codeword0)
        codeword0 = list(codeword0)

        codeword1 = [[line[e[0],v] for v in rows[e[0]] if e[1] != v] for e in lines]
        max_codeword1_len = max(len(c) for c in codeword1)
        codeword1_padded = [np.pad(a, (0, max_codeword1_len - len(a)), 'constant', constant_values=-1).astype(np.int32) for a in codeword1]
        codeword1_padded_reshaped = np.reshape(codeword1_padded, max_codeword1_len*num_edges_tanner, order='C')

        codeword2 = [[line[c,e[1]] for c in cols[e[1]] if e[0] != c] for e in lines]
        max_codeword2_len = max(len(c) for c in codeword2)
        codeword2_padded = [np.pad(a, (0, max_codeword2_len - len(a)), 'constant', constant_values=-1).astype(np.int32) for a in codeword2]
        codeword2_padded_reshaped = np.reshape(codeword2_padded, max_codeword2_len*num_edges_tanner, order='C')

        codeword3 = [[line[c,e[1]] for c in cols[e[1]] if e[0] != c] for lineno, e in enumerate(lines)]
        max_codeword3_len = max(len(c) for c in codeword3)
        codeword3_padded = [np.pad(a, (0, max_codeword3_len - len(a)), 'constant', constant_values=-1).astype(np.int32) for a in codeword3]
        codeword3_padded_reshaped = np.reshape(codeword3_padded, max_codeword3_len*num_edges_tanner, order='C')

        codeword4 = [[line[c, e+1] for c in cols[e+1]] for e in codeword0]
        max_codeword4_len = max(len(c) for c in codeword4)
        codeword4_padded = [np.pad(a, (0, max_codeword4_len - len(a)), 'constant', constant_values=-1).astype(np.int32) for a in codeword4]
        codeword4_padded_reshaped = np.reshape(codeword4_padded, max_codeword4_len*codeword_len, order='C')

        print(len(codeword1), len(codeword1_padded_reshaped), max_codeword1_len)
        print(len(codeword2), len(codeword2_padded_reshaped), max_codeword2_len)
        print(len(codeword3), len(codeword3_padded_reshaped), max_codeword3_len)
        print(len(codeword3), len(codeword4_padded_reshaped), max_codeword4_len)

        for current in lv_series:
#            x = tf.tanh(.5*(tf.gather(wv * current, codeword) + tf.einsum('mn,n->n',tf.multiply(wee, filter_matrix), next_input)))
            print(next_input)
            print('current', current)
            print('wv', wv)

            x = tf.multiply(tf.gather(wv, codeword), tf.gather(current, codeword, axis=1))
            m = tf.gather(abc, [c if c != -1 else 0 for c in codeword3_padded_reshaped])
            m = tf.multiply(m, [1 if a != -1 else 0 for a in codeword3_padded_reshaped])
            m = tf.reshape(m, [num_edges_tanner, max_codeword3_len])
            print('m', m)
            e = tf.gather(next_input, [c if c != -1 else 0 for c in codeword2_padded_reshaped], axis=1)
            print('be', e)
            e = tf.multiply(e, [1 if a != -1 else 0 for a in codeword2_padded_reshaped])
            e = tf.reshape(e, [-1,num_edges_tanner, max_codeword2_len])
            print('e', e)
            s = tf.multiply(m,e)
            print('s', s)
            s = tf.reduce_sum(s,axis=2)

            print('x', x)
            print('m', m)
            print(current)
            print(wv)

            x = tf.tanh(.5*(x + s))
            print('x', x)

            g = tf.gather(x, [c if c != -1 else 0 for c in codeword1_padded_reshaped], axis=1)
            g = tf.multiply(g, [1 if a != -1 else 0 for a in codeword1_padded_reshaped])
            g = tf.add(g, [0 if a != -1 else 1 for a in codeword1_padded_reshaped])
            g = tf.reshape(g, [-1,num_edges_tanner, max_codeword1_len])
            print('g', g)
            next_input = 2*tf.atanh(tf.reduce_prod(g, axis=2))
            print('next', next_input)
            print(count)
            count += 1

        abc1 = tf.Variable(np.random.rand(len(sp_indices2)), name='var3', dtype=tf.float32)
        wwv = tf.Variable(np.random.rand(codeword_len), name='var1',dtype=tf.float32)

        print('current', current)
        x_out = tf.multiply(tf.gather(wwv, codeword0), tf.gather(current, codeword0, axis=1))
        m_out = tf.gather(abc1, [c if c != -1 else 0 for c in codeword4_padded_reshaped])
        m_out = tf.multiply(m_out, [1 if a != -1 else 0 for a in codeword4_padded_reshaped])
        m_out = tf.reshape(m_out, [codeword_len, max_codeword4_len])
        print('m_out', m_out)
        e_out = tf.gather(next_input, [c if c != -1 else 0 for c in codeword4_padded_reshaped], axis=1)
        print('be', e_out)
        e_out = tf.multiply(e_out, [1 if a != -1 else 0 for a in codeword4_padded_reshaped])
        e_out = tf.reshape(e_out, [-1,codeword_len, max_codeword4_len])
        print('e_out', e_out)
        s_out = tf.multiply(m_out,e_out)
        print('s_out1', s_out)
        s_out = tf.reduce_sum(s_out,axis=2)
        print('x_out', x_out)
        print('m_out', m_out)
        print('s_out', s_out)


        output = x_out + s_out

#        print('ni', next_input)
#        mm = tf.sparse_tensor_dense_matmul(wwee, tf.reshape(next_input, [num_edges_tanner, -1]))
#        print('mm',mm)
#        output = tf.reshape(wwv * current + tf.transpose(mm),[batch_size, codeword_len])
#        print('out', output)

        return lv, init_input, tf.nn.sigmoid(output), output, num_edges_tanner

def main(argv):
    lv, x, out, logits, num_edges_tanner = create_model(unwrap)
    loss = tf.reduce_mean(tf.losses.sigmoid_cross_entropy(np.zeros([batch_size, codeword_len]), logits))
    global_step = tf.train.get_or_create_global_step()
    optmizer = tf.train.RMSPropOptimizer(learning_rate=0.003)
    train_op = optmizer.minimize(loss, global_step=global_step)

    hooks=[ tf.train.StopAtStepHook(last_step=FLAGS.train_steps)]

    i = 0
    with tf.train.MonitoredTrainingSession(checkpoint_dir=FLAGS.model_dir, save_checkpoint_steps=FLAGS.checkpoint_steps,hooks=hooks,save_checkpoint_secs=None) as sess:
        print('initializing...')
#        sess.run(tf.global_variables_initializer())

        print('training...')
        while not sess.should_stop():

            _, l = sess.run([train_op, loss], feed_dict={lv: -(.5*variance**2)/(np.ones([batch_size, unwrap, codeword_len]) + np.random.normal(0,variance,[batch_size, unwrap,codeword_len])), x: np.zeros([batch_size, num_edges_tanner])})
            if i % 100 == 0:
                msgNoise = [[1 if b == 1 else -1 for b in msgEnc] for k in range(unwrap)] + np.random.normal(0,variance,[batch_size, unwrap,codeword_len])
                o = sess.run(out, feed_dict={lv: (.5*variance**2)/(msgNoise), x: np.zeros([batch_size, num_edges_tanner])})
                msgRx = [1 if b > .5 else 0 for b in o[0]]
                msgRxNoise = [1 if b > .5 else 0 for b in msgNoise[-1][0]]
                hd = sum([1 for a,b in zip(msgRx[0:len(msgTx)], msgTx) if a != b])
                hd1 = sum([1 for a,b in zip(msgRxNoise[0:len(msgTx)], msgTx) if a != b])
                print('distance: ', hd, hd1)
                print(l)
            i += 1

if __name__ == "__main__":
    tf.app.run()
