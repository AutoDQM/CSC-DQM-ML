from __future__ import division, print_function, absolute_import
import sys, os
try:
    sys.path.remove('/home/users/bemarsh/.local/lib/python2.7/site-packages/matplotlib-1.4.3-py2.7-linux-x86_64.egg')
except:
    pass
import cPickle as pickle
import time
import ROOT
ROOT.gROOT.SetBatch(1)
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import utils
from histDefs import histDefs as hists

indir = "/nfs-6/userdata/bemarsh/CSC_DQM/Run2017/SingleMuon/"

min_train_entries = 10000
learning_rate = 0.01
num_steps=5000
init_param_widths = [0.1, 0.5, 0.7, 0.7, 1.0, 1.0, 1.0]

display_step = 1000

# hists = [("recHits","hRHTimingAnodem22")]

for dname,hname in hists:
    print("Training {0}/{1}".format(dname, hname))

    pkl_name = "hdata_pickles/{0}_{1}.pkl".format(dname, hname)
    if not os.path.exists(pkl_name):
        hstr = "DQMData/Run {{}}/CSC/Run summary/CSCOfflineMonitor/{0}/{1}".format(dname,hname)
        harray, good_rows, runs, n_entries = utils.readHistsFromFiles(indir, hstr, min_entries=min_train_entries)
        pickle.dump((harray, good_rows, runs, n_entries), open(pkl_name, 'wb'))
    else:
        harray, good_rows, runs, n_entries = pickle.load(open(pkl_name, 'rb'))

    nbins = good_rows.size
    n_samples = harray.shape[0]

    if nbins==0:
        print("ERROR: this histogram is always empty! Skipping.")
        continue

    print("{0} samples with sufficient statistics".format(n_samples))

    tf.set_random_seed(int(time.time()))
    
    hidden_layers = [nbins//2, 3]
    
    minloss = 999.
    mindict = {}
    for width in init_param_widths:
        print("Trying width", width)
        # tf Graph input (only pictures)
        X = tf.placeholder("float", [None, nbins])

        layers = [nbins] + hidden_layers
        weights = {}
        for i in range(len(layers)-1):
            weights["encoder_h"+str(i)] = tf.Variable(tf.random_normal([layers[i], layers[i+1]], stddev=width))
            weights["decoder_h"+str(i)] = tf.Variable(tf.random_normal([layers[-i-1], layers[-i-2]], stddev=width))
            
        biases = {}
        for i in range(len(layers)-1):
            biases["encoder_b"+str(i)] = tf.Variable(tf.random_normal([layers[i+1]], stddev=width))
            biases["decoder_b"+str(i)] = tf.Variable(tf.random_normal([layers[-i-2]], stddev=width))

        # Building the encoder
        def encoder(x):
            ls = [x]
            for i in range(len(layers)-1):
                ls.append(tf.nn.sigmoid(tf.add(tf.matmul(ls[i], weights['encoder_h'+str(i)]),
                                               biases['encoder_b'+str(i)])))
            return ls[-1]

        # Building the decoder
        def decoder(x):
            ls = [x]
            for i in range(len(layers)-2):
                ls.append(tf.nn.sigmoid(tf.add(tf.matmul(ls[i], weights['decoder_h'+str(i)]),
                                               biases['decoder_b'+str(i)])))
                ls.append(tf.nn.sigmoid(tf.add(tf.matmul(ls[len(layers)-2], weights['decoder_h'+str(len(layers)-2)]),
                                               biases['decoder_b'+str(len(layers)-2)])))
            return ls[-1]

        # Construct model
        encoder_op = encoder(X)
        decoder_op = decoder(encoder_op)
        
        # Prediction
        y_pred = decoder_op
        # Targets (Labels) are the input data.
        y_true = X
    
        # Define loss and optimizer, minimize the squared error
        loss = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
        optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)

        # Initialize the variables (i.e. assign their default value)
        init = tf.global_variables_initializer()

        # Start Training
        # Start a new TF session
        with tf.Session() as sess:
            
            sess.run(init)

            for i in range(1, num_steps+1):

                # nsamples is small enough to just train over entire dataset each step
                batch_x = harray
            
                # do the training step
                _, l = sess.run([optimizer, loss], feed_dict={X:batch_x})

                # Display logs per step
                if i % display_step == 0 or i == 1:
                    print('Step %i: Minibatch Loss: %f' % (i, l))

            weights_asnp = {}
            biases_asnp = {}

            for key in weights:
                weights_asnp[key] = sess.run(weights[key])
            for key in biases:
                biases_asnp[key] = sess.run(biases[key])

            reconstructed = sess.run(decoder_op, feed_dict={X: harray})
            sses = np.sqrt(np.sum((reconstructed-harray)**2, axis=1))

            outname = "out_pickles/{0}_{1}.pkl".format(dname,hname)
            outdict = { "good_rows" : good_rows,
                        "layer_sizes": layers,
                        "final_loss": l,
                        "weights" : weights_asnp,
                        "biases" : biases_asnp,
                        "sse_array" : sses,
                        "thresh_1pct": np.percentile(sses, 99),
                        "thresh_2pct": np.percentile(sses, 98),
                        "thresh_5pct": np.percentile(sses, 95),
                        "thresh_10pct": np.percentile(sses, 90),
                        }

            if l < minloss:
                minloss = l
                mindict = outdict

    pickle.dump(mindict, open(outname, 'wb'))
