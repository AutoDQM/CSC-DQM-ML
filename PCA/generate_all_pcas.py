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
    harray, good_bins, _, _ = utils.GetHistData(dname, hname, year=2017, entry_cut=10000, force_reload=False)
    pca = PCA()
    pca.fit(harray)
    ixf = pca.inverse_transform(np.append(pca.transform(harray)[:,:1], np.zeros((harray.shape[0], harray.shape[1]-1)), axis=1))
    sses1 = np.sqrt(np.sum((ixf-harray)**2, axis=1))
    ixf = pca.inverse_transform(np.append(pca.transform(harray)[:,:2], np.zeros((harray.shape[0], harray.shape[1]-2)), axis=1))
    sses2 = np.sqrt(np.sum((ixf-harray)**2, axis=1))
    ixf = pca.inverse_transform(np.append(pca.transform(harray)[:,:3], np.zeros((harray.shape[0], harray.shape[1]-3)), axis=1))
    sses3 = np.sqrt(np.sum((ixf-harray)**2, axis=1))
    
    d = {}
    d["pca"] = pca
    d["good_bins"] = good_bins
    d["sses_1comp"] = {}
    d["sses_1comp"]["1pct"] = np.percentile(sses1, 99)
    d["sses_1comp"]["2pct"] = np.percentile(sses1, 98)
    d["sses_1comp"]["5pct"] = np.percentile(sses1, 95)
    d["sses_1comp"]["10pct"] = np.percentile(sses1, 90)
    d["sses_2comp"] = {}
    d["sses_2comp"]["1pct"] = np.percentile(sses2, 99)
    d["sses_2comp"]["2pct"] = np.percentile(sses2, 98)
    d["sses_2comp"]["5pct"] = np.percentile(sses2, 95)
    d["sses_2comp"]["10pct"] = np.percentile(sses2, 90)
    d["sses_3comp"] = {}
    d["sses_3comp"]["1pct"] = np.percentile(sses3, 99)
    d["sses_3comp"]["2pct"] = np.percentile(sses3, 98)
    d["sses_3comp"]["5pct"] = np.percentile(sses3, 95)
    d["sses_3comp"]["10pct"] = np.percentile(sses3, 90)

    pickle.dump(d, open(outdir + "/{0}_{1}.pkl".format(dname, hname), 'wb'))
