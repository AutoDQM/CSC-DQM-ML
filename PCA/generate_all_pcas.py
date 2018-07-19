import sys, os
sys.path.append("../common")
import numpy as np
from sklearn.decomposition import PCA
import utils
from histDefs import histDefs
import cPickle as pickle

outdir = "out_pickles/2017"
os.system("mkdir -p "+outdir)
for dname, hname in histDefs:
    print dname, hname
    harray, _, _, _ = utils.GetHistData(dname, hname, year=2017, entry_cut=10000)
    pca = PCA()
    pca.fit(harray)
    ixf = pca.inverse_transform(pca.transform(harray))
    sses = np.sqrt(np.sum((ixf-harray)**2, axis=1))
    
    d = {}
    d["pca"] = pca
    d["sses"] = sses

    pickle.dump(d, open(outdir + "/{0}_{1}.pkl".format(dname, hname), 'wb'))
