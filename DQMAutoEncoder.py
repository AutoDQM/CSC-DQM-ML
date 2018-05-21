from __future__ import division, print_function, absolute_import
import numpy as np
import tensorflow as tf

class DQMAutoEncoder:
    def __init__(self, layer_sizes, learning_rate=0.01, num_iters=5000, init_weight_spread=1.0, batch_size=None):

        self.layer_sizes = layer_sizes
        self.learning_rate = 0.01
        self.num_iters = num_iters
        self.batch_size = batch_size

        self.X = tf.placeholder("float", [None, layer_sizes[0]])

        self.weights = {}
        self.biases = {}

        self.initializeWeights(init_weight_spread)
        self.encoder = self.getEncoder(self.X)
        self.decoder = self.getDecoder(self.encoder)
                
    def initializeWeights(self, weight_spread):
        for i in range(len(self.layer_sizes)-1):
            # self.weights["encoder_"+str(i)] = tf.Variable(tf.random_normal([self.layer_sizes[i], self.layer_sizes[i+1]], 
            #                                                                stddev=weight_spread))
            # self.weights["decoder_"+str(i)] = tf.Variable(tf.random_normal([self.layer_sizes[-i-1], self.layer_sizes[-i-2]], 
            #                                                                stddev=weight_spread))
            a = 4.0*np.sqrt(6.0/(self.layer_sizes[i] + self.layer_sizes[i-1]))
            self.weights["encoder_"+str(i)] = tf.Variable(tf.random_uniform([self.layer_sizes[i], self.layer_sizes[i+1]], -a, a))
            a = 4.0*np.sqrt(6.0/(self.layer_sizes[-i-1] + self.layer_sizes[-i-2]))
            self.weights["decoder_"+str(i)] = tf.Variable(tf.random_uniform([self.layer_sizes[-i-1], self.layer_sizes[-i-2]], -a, a))
        for i in range(len(self.layer_sizes)-1):
            # self.biases["encoder_"+str(i)] = tf.Variable(tf.random_normal([self.layer_sizes[i+1]], stddev=weight_spread))
            # self.biases["decoder_"+str(i)] = tf.Variable(tf.random_normal([self.layer_sizes[-i-2]], stddev=weight_spread))
            self.biases["encoder_"+str(i)] = tf.Variable(tf.zeros([self.layer_sizes[i+1]]))
            self.biases["decoder_"+str(i)] = tf.Variable(tf.zeros([self.layer_sizes[-i-2]]))

    def getEncoder(self, x):
        ls = [x]
        for i in range(len(self.layer_sizes)-1):
            ls.append(tf.nn.sigmoid(tf.add(tf.matmul(ls[i], self.weights['encoder_'+str(i)]),
                                           self.biases['encoder_'+str(i)])))
        return ls[-1]
        
    def getDecoder(self, x):
        ls = [x]
        for i in range(len(self.layer_sizes)-1):
            ls.append(tf.nn.sigmoid(tf.add(tf.matmul(ls[i], self.weights['decoder_'+str(i)]),
                                           self.biases['decoder_'+str(i)])))
        return ls[-1]

    def train(self, sess, X, batch_size, output_every=None):
        self.y_pred = self.decoder
        self.y_true = self.X
        self.loss = tf.reduce_mean(tf.pow(self.y_true - self.y_pred, 2))
        self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)

        sess.run(tf.global_variables_initializer())

        for i in range(1, self.num_iters+1):
            if self.batch_size is None:
                batch_X = X
            else:
                batch_X = X[np.random.choice(np.arange(X.shape[0]), batch_size, replace=False), :]

            _, l = sess.run([self.optimizer, self.loss], feed_dict={self.X:batch_X})

            if output_every is not None and (i==1 or i%output_every==0):
                print('Step {0}: Batch loss: {1}'.format(i, l))

        return l

    def run(self, sess, X, justEncoder=False):
        if justEncoder:
            res = sess.run(self.encoder, feed_dict={self.X: X})
        else:
            res = sess.run(self.decoder, feed_dict={self.X: X})
        return res

