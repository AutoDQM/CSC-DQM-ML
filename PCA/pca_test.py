import os,sys,json
sys.path.append("../common")
import cPickle as pickle
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import ROOT
ROOT.gROOT.SetBatch(1)
import utils

drawPlots=False
run_info = json.load(open('../../run_info.json'))

# dname, hname = "Digis", "hWireTBin_p11b"
# dname, hname = "recHits", "hRHTimingm22"
# dname, hname = "recHits", "hRHTimingAnodep21"
# dname, hname = "recHits", "hRHSumQm11a"
# dname, hname = "Segments", "hSnSegments"
# dname, hname = "PedestalNoise", "hStripPedMEm21"
# dname, hname = "Segments", "hSTimeCathode"
dname, hname = "Segments", "hSTimeCombined"
# dname, hname = "Segments", "hSGlobalTheta"
# dname, hname = "Resolution", "hSResidp12"
# dname, hname = "BXMonitor", "hCLCTL1A"
# dname, hname = "recHits", "hRHnrechits"

harray, _, runs, _ = utils.GetHistData(dname, hname, entry_cut=10000)
harray_all, _, runs_all, _ = utils.GetHistData(dname, hname, entry_cut=1)

harray18, _, runs18, _ = utils.GetHistData(dname, hname, entry_cut=10000, year=2018)
harray18_all, _, runs18_all, _ = utils.GetHistData(dname, hname, entry_cut=1, year=2018)

harray = np.append(harray, harray18, axis=0)
harray_all = np.append(harray_all, harray18_all, axis=0)
runs = np.append(runs, runs18)
runs_all = np.append(runs_all, runs18_all)

try:
    bad_runs = [int(x) for x in open("../../tf_test/bad_runs/{0}_{1}".format(dname, hname)).readlines()]
except:
    bad_runs = []

lumis_all =[]
for run in runs_all:
    if str(run) not in run_info:
        lumis_all.append(0)
        continue
    A = run_info[str(run)]["Initial Lumi"]
    B = run_info[str(run)]["Ending Lumi"]
    if A<0.1 or B<0.1:
        lumis_all.append(0)
    elif A==B:
        lumis_all.append(A)
    else:
        lumis_all.append((A-B)/np.log(A/B))
lumis =[]
for run in runs:
    if str(run) not in run_info:
        lumis.append(0)
        continue
    A = run_info[str(run)]["Initial Lumi"]
    B = run_info[str(run)]["Ending Lumi"]
    if A<0.1 or B<0.1:
        lumis.append(0)
    elif A==B:
        lumis.append(A)
    else:
        lumis.append((A-B)/np.log(A/B))
lumis = np.array(lumis)
lumis_all = np.array(lumis_all)

pca = Pipeline([
        ("scaler", StandardScaler(with_mean=False, with_std=False)),
        ("pca", PCA(n_components=3)),
        ])

pca.fit(harray)
varrat = pca.named_steps["pca"].explained_variance_ratio_
for i,v in enumerate(varrat[:10]):
    print "Component {0}: {1:.1f}%".format(i, 100*v)
    

xformed = pca.transform(harray)
ixformed = pca.inverse_transform(xformed)
center = np.mean(xformed, axis=0)

xformed_all = pca.transform(harray_all)
ixformed_all = pca.inverse_transform(xformed_all)

sses = np.sqrt(np.sum((ixformed_all-harray_all)**2, axis=1))
dists = np.linalg.norm(xformed_all - np.tile(center, xformed_all.shape[0]).reshape(-1, center.size), axis=1)
print np.mean(sses)

if drawPlots:
    outdir = "/home/users/bemarsh/public_html/dump/DQMML_test/pca_{0}_{1}".format(dname, hname)
    os.system("mkdir -p "+outdir)
    os.system("mkdir -p "+outdir+"/aux")
    os.system("cp ~/scripts/index.php "+outdir)
    os.system("cp ~/scripts/index.php "+outdir+"/aux")
    nbins = harray.shape[1]
    for i in range(len(runs_all)):

        ho = ROOT.TH1D("ho_"+str(i), "", nbins, 0, nbins)
        hr = ROOT.TH1D("hr_"+str(i), "", nbins, 0, nbins)
    
        for ib in range(nbins):
            ho.SetBinContent(ib+1, harray_all[i, ib])
            hr.SetBinContent(ib+1, ixformed_all[i, ib])

        c = ROOT.TCanvas()
        
        ho.SetLineColor(ROOT.kBlue)
        ho.SetLineWidth(4)
        hr.SetLineColor(ROOT.kRed)
        hr.SetLineWidth(4)
        hr.SetLineStyle(9)
        
        ho.Draw("HIST")
        hr.Draw("SAME HIST")
        
        text = ROOT.TLatex()
        text.SetTextFont(62)
        text.SetTextColor(ROOT.kBlue)
        text.SetTextSize(0.06)
        text.DrawLatexNDC(0.7, 0.65, "{0:.5f}".format(sses[i]))
        text.DrawLatexNDC(0.7, 0.60, "{0:.3f}".format(xformed_all[i,0]))
        text.DrawLatexNDC(0.7, 0.55, "{0:.3f}".format(xformed_all[i,1]))
        text.DrawLatexNDC(0.7, 0.50, "{0:.3f}".format(xformed_all[i,2]))
        text.DrawLatexNDC(0.7, 0.45, "{0:.0f}".format(lumis_all[i]))
    
        c.SaveAs("{0}/{1}.png".format(outdir, runs_all[i]))
    

fig1 = plt.figure(figsize=plt.figaspect(0.5))
ax3d = fig1.add_subplot(121, projection='3d')
ax3d.scatter(xformed[:,0], xformed[:,1], xformed[:,2], c=runs, cmap='Spectral', vmin=np.amin(runs), vmax=np.amax(runs), linewidth=0.3)
ax3d.set_xlabel("x1")
ax3d.set_ylabel("x2")
ax3d.set_zlabel("x3")

ax2d = fig1.add_subplot(122)
# ax2d.scatter(xformed[lumis>0,0], xformed[lumis>0,1], c=lumis[lumis>0], cmap='Spectral', vmin=np.amin(lumis), vmax=np.amax(lumis), linewidth=0.3)
ax2d.scatter(xformed[:,0], xformed[:,1], c=runs, cmap='Spectral', vmin=np.amin(runs), vmax=np.amax(runs), linewidth=0.3)
ax2d.set_xlabel("x1")
ax2d.set_ylabel("x2")

fig2 = plt.figure()
ax0 = fig2.add_subplot(111)
dim1 = pca.inverse_transform([[1, 0, 0]])
dim2 = pca.inverse_transform([[0, 1, 0]])
dim3 = pca.inverse_transform([[0, 0, 1]])
xvals = np.zeros(2*dim1.size)
yvals0 = np.zeros(2*dim1.size)
yvals1 = np.zeros(2*dim1.size)
yvals2 = np.zeros(2*dim1.size)
yvals3 = np.zeros(2*dim1.size)
yvalst = np.zeros(2*dim1.size)
xvals[0::2] = np.arange(dim1.size)
xvals[1::2] = np.arange(1, dim1.size+1)
yvals0[0::2] = pca.named_steps["scaler"].inverse_transform(pca.named_steps["pca"].mean_)
yvals1[0::2] = dim1[0,:]*xformed[0,0]
yvals2[0::2] = dim2[0,:]*xformed[0,1]
yvals3[0::2] = dim3[0,:]*xformed[0,2]
yvalst[0::2] = ixformed[0,:]
yvals0[1::2] = pca.named_steps["scaler"].inverse_transform(pca.named_steps["pca"].mean_)
yvals1[1::2] = dim1[0,:]*xformed[0,0]
yvals2[1::2] = dim2[0,:]*xformed[0,1]
yvals3[1::2] = dim3[0,:]*xformed[0,2]
yvalst[1::2] = ixformed[0,:]
hist0, = ax0.plot(xvals, yvals0, 'k-')
hist1, = ax0.plot(xvals, yvals1, 'b-')
hist2, = ax0.plot(xvals, yvals2, 'r-')
hist3, = ax0.plot(xvals, yvals3, 'g-')
histt, = ax0.plot(xvals, yvalst, 'r--', linewidth=2)

plt.figure()
good_scores = []
bad_scores = []
scores = sses
for i in range(len(runs_all)):
    if runs_all[i] in bad_runs:
        bad_scores.append(scores[i])
    else:
        good_scores.append(scores[i])
good_scores = np.array(good_scores)
bad_scores = np.array(bad_scores)
fpr = 1.0*np.sum(good_scores >= np.amin(bad_scores)) / good_scores.size if bad_scores.size>0 else 0.0
print("FPR:", fpr)

pcts = [1,2,5,10]
threshs = [np.percentile(scores, 100-pct) for pct in pcts]
plt.hist(good_scores, bins=50, range=(0, np.amax(scores)+0.05), histtype='stepfilled', color='g', alpha=0.5)
plt.hist(bad_scores, bins=50, range=(0, np.amax(scores)+0.05), histtype='stepfilled', color='r', alpha=0.5)
for t in threshs:
    plt.plot([t]*2, [0.1,1000], 'k--')
plt.gca().set_yscale('log')
plt.gca().set_ylim(0.1,1000)

def onclick(event):
    if event.xdata==None:
        return
    if event.inaxes is not ax2d:
        return
    # print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
    #       ('double' if event.dblclick else 'single', event.button,
    #        event.x, event.y, event.xdata, event.ydata))
    ixf = pca.inverse_transform([[event.xdata, event.ydata, 0]])

    yvals1[0::2] = dim1[0,:]*event.xdata
    yvals1[1::2] = dim1[0,:]*event.xdata
    yvals2[0::2] = dim2[0,:]*event.ydata
    yvals2[1::2] = dim2[0,:]*event.ydata
    yvals3[0::2] = dim3[0,:]*0
    yvals3[1::2] = dim3[0,:]*0
    yvalst[0::2] = ixf[0, :]
    yvalst[1::2] = ixf[0, :]

    hist1.set_ydata(yvals1)
    hist2.set_ydata(yvals2)
    hist3.set_ydata(yvals3)
    histt.set_ydata(yvalst)

    fig2.canvas.draw()
    fig2.canvas.flush_events()

cid = fig1.canvas.mpl_connect('button_press_event', onclick)

plt.show()
