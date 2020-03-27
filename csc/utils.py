import os, sys
sys.path.append("..")
import ROOT as r
import numpy as np
import glob as glob
import cPickle as pickle
import json
from dqmml.HistCollection import *

# read directly from DQM files
def read_hists_from_files(indir, hpath, runs_to_get=None, max_bins=None, norm_cut=None):
    runs = []
    hists = []
    if runs_to_get==None:
        fnames = glob.glob(indir+"/*.root")
    else:
        fnames = ["{0}/{1}.root".format(indir, run) for run in runs_to_get]
    title = None
    for fname in fnames:
        run = int(fname.split("/")[-1].split(".")[0])
        # print run
        fin = r.TFile(fname)

        h = fin.Get(hpath.format(run))
        if norm_cut is None or h.Integral() >= norm_cut:
            nbins = h.GetNbinsX() if max_bins==None else min(h.GetNbinsX(), max_bins)
            hists.append([h.GetBinContent(i) for i in range(1, nbins+1)])
            runs.append(run)

        if title is None:
            title = h.GetTitle()

        fin.Close()

    # make rows even length if jagged
    lens = [len(row) for row in hists]
    maxlen = np.amax(lens)
    if maxlen != np.amin(lens):
        for i in range(len(hists)):
            hists[i] += [0]*(maxlen - len(hists[i]))

    hists = np.array(hists)
    if hists.size==0:
        return hists, np.array([])

    return hists, np.array(runs), title

# get array of histogram contents
def load_hist_data(dname, hname, pkl_dir="data/{year}", year=2017, raw_dir="/nfs-7/userdata/bemarsh/CSC_DQM/Run{year}/SingleMuon", 
                   force_reload=False, max_bins=None, lumi_json=None, norm_cut=None):
    pkl_dir = pkl_dir.format(year=year)
    pkl_name = os.path.join(pkl_dir, "{0}_{1}.pkl".format(dname, hname))
    if not force_reload and os.path.exists(pkl_name):
        with open(pkl_name, 'rb') as fid:
            hc = pickle.load(fid)
        if norm_cut is not None:
            orig_harray = hc.hist_cleaner.restore_bad_bins(hc.hdata*hc.norms.reshape(-1,1))
            mask = hc.norms>=norm_cut
            ei = hc.extra_info
            ei["runs"] = ei["runs"][mask]
            ei["lumis"] = ei["lumis"][mask]
            hc = HistCollection(orig_harray[mask,:], extra_info=hc.extra_info)
        return hc
    else:
        os.system("mkdir -p "+pkl_dir)
        hstr = "DQMData/Run {{}}/CSC/Run summary/CSCOfflineMonitor/{0}/{1}".format(dname, hname)
        indir = raw_dir.format(year=year)
        harray, runs, title = read_hists_from_files(indir, hstr, max_bins=max_bins, norm_cut=norm_cut)
        extra_info = {"runs":runs}
        extra_info["title"]  = title
        if lumi_json is not None:
            with open(lumi_json, 'rb') as fid:
                ri = json.load(fid)
            lumis = []
            for run in runs:
                if str(run) in ri:
                    A = ri[str(run)]["Initial Lumi"]
                    B = ri[str(run)]["Ending Lumi"]
                    if A<0.1 or B<0.1:
                        lumis.append(0)
                    elif A==B:
                        lumis.append(A)
                    else:
                        lumis.append((A-B)/np.log(A/B))
                else:
                    lumis.append(0)
            extra_info["lumis"] = np.array(lumis)
        hc = HistCollection(harray, extra_info=extra_info)
        with open(pkl_name, 'wb') as fid:
            pickle.dump(hc, fid, protocol=-1)
        return hc

