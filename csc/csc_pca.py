import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
sys.path.append("..")
from dqmml.HistCollection import *
from dqmml.DQMPCA import *
import utils

# Which CSC histogram should we use?
dname, hname = "Segments", "hSTimeCombined"
# dname, hname = "Digis", "hWireTBin_p11b"

# Load in histogra data. If the pickle file doesn't exist yet, load from raw DQM ROOT files and save
hc = utils.load_hist_data(dname, hname, pkl_dir="data/test", lumi_json="run_info.json")

# create a DQMPCA object and train on the histogram data
# Use only histograms with >10000 entries
pca = DQMPCA(norm_cut=10000, sse_ncomps=(1,2,3))
pca.fit(hc)

# Compute the number of components to use for SSE/score calculation (lowest # that explains >95% of variance)
evr = pca.explained_variance_ratio
ncomp = min(np.argmax(np.cumsum(evr)>0.95)+1, 3)

# Compute the transformed and inverse transformed histograms, and compute SSEs/scores
xf = pca.transform(hc)
ixf = pca.inverse_transform(xf, n_components=ncomp)
sses = pca.sse(hc, n_components=ncomp)
scores = pca.score(hc, n_components=ncomp)

# Throw away low stats histograms
goodrows = hc.norms>10000
xf = xf[goodrows, :]
ixf = ixf[goodrows, :]
sses = sses[goodrows]
scores = scores[goodrows]
lumis = hc.extra_info["lumis"][goodrows]
runs = hc.extra_info["runs"][goodrows]

# 3D plot of first 3 transformed components, colored by run #
fig1 = plt.figure(figsize=plt.figaspect(0.5))
ax3d = fig1.add_subplot(121, projection='3d')
ax3d.scatter(xf[:,0], xf[:,1], xf[:,2], c=runs, cmap='Spectral', vmin=np.amin(runs), vmax=np.amax(runs), linewidth=0.3)
ax3d.set_xlabel("x1")
ax3d.set_ylabel("x2")
ax3d.set_zlabel("x3")

# 2D plot of first 2 transformed components
ax2d = fig1.add_subplot(122)
# ax2d.scatter(xf[lumis>0,0], xf[lumis>0,1], c=lumis[lumis>0], cmap='Spectral', vmin=np.amin(lumis), vmax=np.amax(lumis), linewidth=0.3)
ax2d.scatter(xf[:,0], xf[:,1], c=runs, cmap='Spectral', vmin=np.amin(runs), vmax=np.amax(runs), linewidth=0.3)
ax2d.set_xlabel("x1")
ax2d.set_ylabel("x2")

# Plot contributions of first 3 components, for an example histogram
fig2 = plt.figure()
ax0 = fig2.add_subplot(111)
dim1 = pca.inverse_transform([[1, 0, 0]], n_components=3)
dim2 = pca.inverse_transform([[0, 1, 0]], n_components=3)
dim3 = pca.inverse_transform([[0, 0, 1]], n_components=3)
xvals = np.zeros(2*dim1.size)
yvals0 = np.zeros(2*dim1.size)
yvals1 = np.zeros(2*dim1.size)
yvals2 = np.zeros(2*dim1.size)
yvals3 = np.zeros(2*dim1.size)
yvalst = np.zeros(2*dim1.size)
xvals[0::2] = np.arange(dim1.size)
xvals[1::2] = np.arange(1, dim1.size+1)
yvals0[0::2] = pca.mean
yvals1[0::2] = dim1[0,:]*xf[0,0]
yvals2[0::2] = dim2[0,:]*xf[0,1]
yvals3[0::2] = dim3[0,:]*xf[0,2]
yvalst[0::2] = ixf[0,:]
yvals0[1::2] = pca.mean
yvals1[1::2] = dim1[0,:]*xf[0,0]
yvals2[1::2] = dim2[0,:]*xf[0,1]
yvals3[1::2] = dim3[0,:]*xf[0,2]
yvalst[1::2] = ixf[0,:]
hist0, = ax0.plot(xvals, yvals0, 'k-')
hist1, = ax0.plot(xvals, yvals1, 'b-')
hist2, = ax0.plot(xvals, yvals2, 'r-')
hist3, = ax0.plot(xvals, yvals3, 'g-')
histt, = ax0.plot(xvals, yvalst, 'r--', linewidth=2)

# Make histogram of scores. Plot "bad runs" in red if these have been defined

try:
    bad_runs = [int(x) for x in open("bad_runs/{0}_{1}".format(dname, hname)).readlines()]
except:
    bad_runs = []

fig3 = plt.figure()
good_scores = []
bad_scores = []
scores = scores
for i in range(len(runs)):
    if runs[i] in bad_runs:
        bad_scores.append(scores[i])
    else:
        good_scores.append(scores[i])
good_scores = np.array(good_scores)
bad_scores = np.array(bad_scores)
fpr = 1.0*np.sum(good_scores >= np.amin(bad_scores)) / good_scores.size if bad_scores.size>0 else 0.0
print("FPR:", fpr)

plt.hist(good_scores, bins=50, range=(0, np.amax(scores)+0.05), histtype='stepfilled', color='g', alpha=0.5)
plt.hist(bad_scores, bins=50, range=(0, np.amax(scores)+0.05), histtype='stepfilled', color='r', alpha=0.5)

# Clicking on the 2D plot of transformed values will update the component histograms in fig2

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
