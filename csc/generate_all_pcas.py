import os, sys
import cPickle as pickle
sys.path.append("..")
from dqmml.HistCollection import *
from dqmml.DQMPCA import *
import utils
from histDefs import histDefs

for dname, hname in histDefs:
    print dname, hname
    hc = utils.load_hist_data(dname, hname)
    pca = DQMPCA()
    pca.fit(hc, norm_cut=10000, sse_ncomps=(1,2,3))

    try:
        os.makedirs("trained_pcas/2017/")
    except OSError:
        pass
    pickle.dump(pca, open("trained_pcas/2017/{0}_{1}.pkl".format(dname, hname), 'wb'))
