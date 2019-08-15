import numpy as np

class HistCleaner(object):
    """ Perform some "cleaning" necessary for feeding into ML algorithms
    Normalizes each histogram to have unit integral, and removes
    bins that are the same for every histogram in the initialization data
    """
    def __init__(self, normalize=True, remove_identical_bins=True):
        self.normalize = normalize
        self.remove_identical_bins= remove_identical_bins
        self.is_fit = False

    def fit(self, hd):
        self.nbins = hd.shape[1]
        # find the "good" bin indices (those that aren't the same in every histogram)
        bad_bins = np.all(hd==np.tile(hd[0,:],hd.shape[0]).reshape(hd.shape), axis=0)
        good_bins = np.logical_not(bad_bins)
        self.bad_bins = np.arange(self.nbins)[bad_bins]
        self.good_bins = np.arange(self.nbins)[good_bins]
        self.n_good_bins = self.good_bins.size
        self.bad_bin_contents = hd[0,self.bad_bins]

        self.is_fit = True

    def _check_fit(self):
        if not self.is_fit:
            raise Exception("Must fit the HistCleaner before calling transform")

    def restore_bad_bins(self, hd):
        self._check_fit()
        init_shape = hd.shape
        if len(init_shape) == 1:
            hd = hd.reshape(1,-1)
        if hd.shape[1] != self.n_good_bins:
            raise Exception("Invalid number of columns")

        ret = np.zeros((hd.shape[0], self.nbins))
        ret[:,self.good_bins] = hd
        ret[:,self.bad_bins] = np.tile(self.bad_bin_contents, hd.shape[0]).reshape(hd.shape[0], self.bad_bins.size)

        if len(init_shape) == 1:
            ret = ret.reshape(ret.size,)
        return ret

    def remove_bad_bins(self, hd):
        self._check_fit() 
        init_shape = hd.shape
        if len(init_shape) == 1:
            hd = hd.reshape(1,-1)
        if hd.shape[1] != self.nbins:
            raise Exception("Invalid number of columns")
        
        ret = hd[:,self.good_bins]
        if len(init_shape) == 1:
            ret = ret.reshape(ret.size,)
        return ret

    def transform(self, hd):
        self._check_fit()
        init_shape = hd.shape
        if len(init_shape)==1:
            hd = hd.reshape(1,-1)
        is_cleaned = False
        if hd.shape[1] != self.nbins:
            if hd.shape[1] == self.n_good_bins:
                is_cleaned = True
            else:
                raise Exception("Invalid shape! Expected {0} or {1} columns, got {2}".format(self.nbins,self.n_good_bins, hd.shape[1]))

        # remove bad bins
        if not is_cleaned and self.remove_identical_bins:
            hd = self.remove_bad_bins(hd)

        # normalize each row
        if self.normalize:
            norms = np.sum(hd, axis=1)
            tile = np.tile(norms, self.n_good_bins).reshape(self.n_good_bins, -1).T
            hd = np.divide(hd, tile, out=np.zeros_like(hd), where=tile!=0)

        if len(init_shape) == 1:
            hd = hd.reshape(hd.size,)
        return hd

class HistCollection(object):
    def __init__(self, hdata, extra_info=None):
        """Initialize the HistCollection
        
        hdata is a 2D array of histogram data
          Each row is a histogram and each column a bin
        extra_info is a dict containing any auxiliary info you want to be stored
          (e.g. extra_info["runs"] could be a list of runs corresponding to each histogram)

        The histograms will be "cleaned" using the HistCleaner class
        """
        self.hdata = np.array(hdata, dtype=float)
        self.nhists = self.hdata.shape[0]
        self.nbins = self.hdata.shape[1]

        self.norms = np.sum(hdata, axis=1)

        self.hist_cleaner = HistCleaner()
        self.hist_cleaner.fit(self.hdata)
        self.hdata = self.hist_cleaner.transform(self.hdata)

        self.extra_info = extra_info

