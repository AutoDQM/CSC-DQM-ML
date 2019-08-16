## CSC-DQM-ML

Collection of ML-based tools for automating **D**ata **Q**uality **M**onitoring with 1D histograms.
Originally made for the CSC subgroup of the Muon DPG at CMS, but attempts have been made to keep things 
general for use in other applications. 
[See here](https://drive.google.com/open?id=18OZjYioG4-xLIwSYC0Kh9Yz1Qobsn5KJ) 
for a set of slides on the topic presented at the 
[2018 MLSE conference](https://events.mcs.cmu.edu/mlse/).

<p align="center">
<img src="./images/scatter_xf.png" alt="scatter plot of a PCA-transformed set of DQM histograms" width="700"/>
 <br><i>3D and 2D projections of a set of PCA-transformed DQM histograms</i>
</p>

Data Quality Monitoring is an important yet time-consuming task for any particle physics experiment.
For the CSC subsystem at CMS (as well as many other subsystems), the usual workflow is as follows:
* For each "run" (multiple per day), automated scripts produce hundreds of plots relating to the performance of the detector.
* A human user manually sifts through all of these plots, comparing against a suitable reference run.
* The user flags any that look unusual and notifies relevant detector experts to diagnose the problem.

Repeating this each day for many runs is tedious and time-consuming for the user.
Hence, there is an ongoing push to automate this process as much as possible using
both statistical and machine learning-based tools.

Here, we have developed a set of tools for **outlier detection** of 1-dimensional DQM histograms. 
The idea is to flag any that look "unusual" compared to previous runs for further inspection by a human user.

This is done primarily through dimensionality reduction. There is typically not much variation in a
given DQM histogram from run to run, so, thinking of each plot as a point in (*nbins*)-dimensional space,
the entire collection can mostly be described by a handful of "components".

### Contents
* `dqmml`
  * Contains main classes for organizing histogram collections and training models to detect outliers.
  * `HistCollection.py` - organize a collection of histograms and "clean" properly (normalize, remove
  bins that are identical between all histograms).
  * `DQMPCA.py` and `DQMAutoEncoder.py` - implementations of PCA (sklearn) and AutoEncoder (tensorflow)
  models for dimensionality reduction.
* `csc`
  * Example implementation of the above, specifically for CSC DQM.
  * `utils.py` - helper functions to load histogram data from a CSC-specific format and store as a `HistCollection`.
  * `data/test` - pickled `HistCollection` objects for two example CSC plots
  * `csc_pca.py` - Full example of loading data, training a `DQMPCA` model, and making some useful plots.
  * `trained_pcas/test` - example pickled trained `DQMPCA` models. The idea would be to load this in the future,
  compare against some new histograms, and flag any above a certain "outlier score".
  
### Example of use
