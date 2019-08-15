import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

class DQMPCA:
    def __init__(self, use_standard_scaler=False):
        if use_standard_scaler:
            self.pca = Pipeline(
                ("scaler", StandardScaler()),
                ("pca", PCA())
                )
        else:
            self.pca = PCA()

    def get_train_subset(self, hcoll, norm_cut=0):
        temp = hcoll.hdata[hcoll.norms>norm_cut, :]
        return temp[:, hcoll.good_bins]

    def fit(self, hcoll, norm_cut=0):
        self.pca.fit(self.get_train_subset(hcoll, norm_cut))
        
    def transform(self, hcoll):
        return self.pca.transform(hcoll.hdata[:, hcoll.good_bins])

    def inverse_transform(self, xf, hcoll, n_components=3):
        trunc = np.zeros(xf.shape)
        trunc[:,:n_components] = xf[:,:n_components]
        ixf = self.pca.inverse_transform(trunc)
        return hcoll.restore_bad_bins(ixf)
