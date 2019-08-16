import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from HistCollection import *

class DQMPCA(object):
    def __init__(self, use_standard_scaler=False):
        """Initialize the DQMPCA

        -use_standard_scalar determines whether to use standard scaling
          (zero mean, unit stdev) before feeding into a PCA. This helps
          for some histograms, but hurts for others
        """
        self.use_standard_scaler = use_standard_scaler
        if use_standard_scaler:
            self.pca = Pipeline(
                ("scaler", StandardScaler()),
                ("pca", PCA())
                )
        else:
            self.pca = PCA()

        self.is_fit = False

    def fit(self, hdata, norm_cut=0, sse_ncomps=None):
        if isinstance(hdata, HistCollection):
            self.hist_cleaner = hdata.hist_cleaner
            cleaned = hdata.hdata
            norms = hdata.norms
        else:
            self.hist_cleaner = HistCleaner()
            hist_cleaner.fit(hdata)
            cleaned = hist_cleaner.transform(hdata)
            norms = np.sum(cleaned, axis=1)

        cleaned = cleaned[norms>norm_cut, :]
        self.pca.fit(cleaned)        
        if sse_ncomps is not None:
            self.sse_cuts = {}
            for ncomp in sse_ncomps:
                self.sse_cuts[ncomp] = []
                sses = self.sse(cleaned, ncomp)
                for pct in np.arange(1,101):
                    self.sse_cuts[ncomp].append(np.percentile(sses, pct))

        self.is_fit = True
    
    def transform(self, hdata):
        """Transform a set of histograms with the trained PCA"""
        if isinstance(hdata, HistCollection):
            cleaned = hdata.hdata
        else:
            cleaned = self.hist_cleaner.transform(hdata)        
        return self.pca.transform(cleaned)
        
    def inverse_transform(self, xf, n_components=3, restore_bad_bins=False):
        xf = np.array(xf)
        trunc = np.zeros((xf.shape[0], self.hist_cleaner.n_good_bins))
        trunc[:,:n_components] = xf[:,:n_components]
        ixf = self.pca.inverse_transform(trunc)
        if not restore_bad_bins:
            return ixf
        else:
            return self.hist_cleaner.restore_bad_bins(ixf)

    def sse(self, hdata, n_components=3):
        if isinstance(hdata, HistCollection):
            cleaned = hdata.hdata
        else:
            cleaned = self.hist_cleaner.transform(hdata)        
        xf = self.transform(cleaned)
        ixf = self.inverse_transform(xf, n_components=n_components)
        return np.sqrt(np.sum((ixf-cleaned)**2, axis=1))
        
    def score(self, hdata, n_components=3):
        if not hasattr(self, "sse_cuts") or n_components not in self.sse_cuts:
            raise Exception("must fit first with {0} in sse_ncomps".format(n_components))
        sse = self.sse(hdata, n_components)
        return np.interp(sse, self.sse_cuts[n_components], np.arange(1,101))

    @property
    def explained_variance_ratio(self):
        if self.use_standard_scaler:
            return self.pca.named_steps["pca"].explained_variance_ratio_
        else:
            return self.pca.explained_variance_ratio_

    @property
    def mean(self):
        if self.use_standard_scaler:
            return self.pca.named_steps["scaler"].inverse_transform(self.pca.named_steps["pca"].mean_)
        else:
            return self.pca.mean_
