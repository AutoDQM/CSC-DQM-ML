from __future__ import division, print_function, absolute_import
import sys, os, psutil
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
from DQMAutoEncoder import DQMAutoEncoder

indir = "/nfs-6/userdata/bemarsh/CSC_DQM/Run2017/SingleMuon/"

min_train_entries = 10000
learning_rate = 0.01
num_iters=5000
# init_param_widths = [0.1, 0.5, 0.7, 0.7, 1.0, 1.0, 1.0]
init_param_widths = [1.0]

display_step = 1000

# hists = [("recHits","hRHTimingAnodem22")]
# hists = [("Digis", "hWireTBin_m11b")]

process = psutil.Process(os.getpid())
print("Mem usage start:", process.memory_info().rss/1e6)

for dname,hname in hists:
    print("Training {0}/{1}".format(dname, hname))

    pkl_name = "hdata_pickles/{0}_{1}.pkl".format(dname, hname)
    if not os.path.exists(pkl_name):
        hstr = "DQMData/Run {{}}/CSC/Run summary/CSCOfflineMonitor/{0}/{1}".format(dname,hname)
        harray, good_rows, runs, n_entries = utils.readHistsFromFiles(indir, hstr, min_entries=min_train_entries)
        pickle.dump((harray, good_rows, runs, n_entries), open(pkl_name, 'wb'))
    else:
        harray, good_rows, runs, n_entries = pickle.load(open(pkl_name, 'rb'))

    print("Mem usage after harray load:", process.memory_info().rss/1e6)

    nbins = good_rows.size
    n_samples = harray.shape[0]

    if nbins==0:
        print("ERROR: this histogram is always empty! Skipping.")
        continue

    print("{0} samples with sufficient statistics".format(n_samples))

    tf.set_random_seed(int(time.time()))
    
    layer_sizes = [nbins, nbins//2, 3]

    minloss = 999.
    mindict = {}
    for width in init_param_widths:
        print("Trying width", width)

        tf.reset_default_graph()

        ae = DQMAutoEncoder(layer_sizes, learning_rate, num_iters, width, None)

        # Start Training
        # Start a new TF session
        with tf.Session() as sess:

            loss = ae.train(sess, harray, 200, display_step)

            weights_asnp = {}
            biases_asnp = {}

            for key in ae.weights:
                weights_asnp[key] = sess.run(ae.weights[key])
            for key in ae.biases:
                biases_asnp[key] = sess.run(ae.biases[key])

            reconstructed = ae.run(sess, harray)

            sses = np.sqrt(np.sum((reconstructed-harray)**2, axis=1))

            outname = "out_pickles/{0}_{1}.pkl".format(dname,hname)
            outdict = { "good_rows" : good_rows,
                        "layer_sizes": layer_sizes,
                        "final_loss": loss,
                        "weights" : weights_asnp,
                        "biases" : biases_asnp,
                        "sse_array" : sses,
                        "thresh_1pct": np.percentile(sses, 99),
                        "thresh_2pct": np.percentile(sses, 98),
                        "thresh_5pct": np.percentile(sses, 95),
                        "thresh_10pct": np.percentile(sses, 90),
                        }

            if loss < minloss:
                minloss = loss
                mindict = outdict

            print("Mem usage after train:", process.memory_info().rss/1e6)

    pickle.dump(mindict, open(outname, 'wb'))
