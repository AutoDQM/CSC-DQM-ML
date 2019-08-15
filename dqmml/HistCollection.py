import numpy as np

class HistCollection:
    def __init__(self, hdata, extra_info=None):
        """Initialize the HistCollection
        
        hdata is a 2D array of histogram data
          Each row is a histogram and each column a bin
        extra_info is a dict containing any auxiliary info you want to be stored
          (e.g. extra_info["runs"] could be a list of runs corresponding to each histogram)
        """
        self.hdata = np.array(hdata, dtype=float)
        self.nhists = self.hdata.shape[0]
        self.nbins = self.hdata.shape[1]

        # normalize each row
        self.norms = np.sum(hdata, axis=1)
        tile = np.tile(self.norms, self.nbins).reshape(self.nbins, self.nhists).T
        self.hdata = np.divide(self.hdata, tile, out=np.zeros_like(self.hdata), where=tile!=0)

        # find the "good" bin indices (those that aren't the same in every histogram)
        bad_bins = np.all(self.hdata==np.tile(self.hdata[0,:],self.nhists).reshape(self.hdata.shape), axis=0)
        good_bins = np.logical_not(bad_bins)
        self.bad_bins = np.arange(self.nbins)[bad_bins]
        self.good_bins = np.arange(self.nbins)[good_bins]
        self.n_good_bins = self.good_bins.size

        self.extra_info = extra_info

    def restore_bad_bins(self, hd):
        if hd.shape[1] != self.n_good_bins:
            raise Exception("Invalid number of columns")

        rest = np.zeros((hd.shape[0], self.nbins))
        rest[:,self.good_bins] = hd
        rest[:,self.bad_bins] = np.tile(self.hdata[0,self.bad_bins], hd.shape[0]).reshape(hd.shape[0], self.bad_bins.size)

        return rest
