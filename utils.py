import ROOT as r
import numpy as np
import glob as glob

def readHistsToArray(fname, bname, n_bins, tname="testTree", min_entries=0, start_bin=1):
    runs = []
    hists = []
    n_ent = []
    fin = r.TFile(fname)
    t = fin.Get(tname)
    for i in range(t.GetEntries()):
        t.GetEntry(i)
        nentries = getattr(t, bname+"_nEntries")
        if nentries < min_entries:
            continue
        h = []
        for j in range(n_bins):
            h.append(getattr(t, bname+"_b{0}".format(j+start_bin)))
        hists.append(h)
        runs.append(t.run)
        n_ent.append(nentries)

    return np.array(hists), runs, n_ent

def readHistsFromFiles(indir, hpath, min_entries=0, runsToGet=None, good_rows=None):
    runs = []
    hists = []
    n_ent = []
    if runsToGet==None:
        fnames = glob.glob(indir+"/*.root")
    else:
        fnames = ["{0}/{1}.root".format(indir, run) for run in runsToGet]
    for fname in fnames:
        run = int(fname.split("/")[-1].split(".")[0])
        fin = r.TFile(fname)

        h = fin.Get(hpath.format(run))
        n_entries = h.GetEntries()
        if n_entries < min_entries:
            continue
        h.Scale(1./h.Integral(0,-1))
        hists.append([h.GetBinContent(i) for i in range(1, h.GetNbinsX()+1)])
        runs.append(run)
        n_ent.append(n_entries)

        fin.Close()

    hists = np.array(hists)
    if hists.size==0:
        return hists, np.array([]), np.array([]), np.array([])
    if good_rows is None:
        bad_rows = np.all(hists==np.tile(hists[0,:],hists.shape[0]).reshape(hists.shape), axis=0)
        good_rows = np.logical_not(bad_rows)
        good_rows = np.arange(good_rows.size)[good_rows]

    hists = hists[:,good_rows]

    return hists, good_rows, runs, n_ent

def evaluateAE(inp, weights, biases):
    nlayers = len(weights) / 2
    res = np.array(inp)
    if len(res.shape)==1:
        res = res.reshape(1, res.size)
    nsamp = res.shape[0]
    for i in range(nlayers):
        res = np.dot(res, weights["encoder_h"+str(i)])
        res += np.tile(biases["encoder_b"+str(i)], nsamp).reshape(res.shape)
        res = 1.0 / (1.0 + np.exp(-res))
    for i in range(nlayers):
        res = np.dot(res, weights["decoder_h"+str(i)])
        res += np.tile(biases["decoder_b"+str(i)], nsamp).reshape(res.shape)
        res = 1.0 / (1.0 + np.exp(-res))

    return res.reshape(inp.shape)

if __name__=="__main__":
    hists,good_rows,runs,n_entries = readHistsFromFiles("/nfs-6/userdata/bemarsh/CSC_DQM/Run2017/SingleMuon/",
                                                        "DQMData/Run {}/CSC/Run summary/CSCOfflineMonitor/recHits/hRHTimingAnodem11a",
                                                        )

    print hists.shape
    print good_rows
    print len(runs)
    print len(n_entries)

