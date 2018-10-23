This repository contains data and code used in the analysis presented in
Muir, Adhikari, and Huterer 2018, "The Covariance of CMB anomalies."

Note that the directory structure and data presented here is arranged for
transparancy, and is NOT in the format that the code was actually run.
Because of that, the code will not work out of the box if you clone the
repository. Rather, it is provided as a reference.

----------------------
PYTHON SCRIPT

The script measure_anomalystats.py does most of the calculation. 
The general outline of this file is:
- utility functions for handling healpix maps (downgrading maps & masks, 
    filename handling, 2pt function measurement, map simulation using synfast, etc)
- functions for measuring features associated with anomalies
- functions for obtaining & handling feature meas. from many realizations
- functions for plotting 1d hists and 2d triangle scatter
- functions for measuring covariance matrix and doing PCA analysis
- the main() function, which has calls to functions for generating data used in analysis. 

----------------------
ANALYSIS PROCEDURE

The anomaly covariance analysis used this code as follows:
- The main() function in measure_anomalystats.py (at the bottom of this file)
  was used to generate &   analyze simulated maps to get anomaly measurements
  for all desired realizations. There's   a hack there, where bool switches
  "if 0:" or "if 1" are used to manually turn off and on parts of the
  calculation (if e.g. only one part needs to be rerun).
  -> The workhorse function used by that script is run_manystats which,
     given a list of stat names  calls the necessary functions to compute
     them for all realizations, and handles file-naming.
- Once the stat data is generated, a jupyter notebook (not included in this
  repo) was used to  analyze data and do plotting. It imports this python
  file as a module, and so uses  some of the covariance calculation and
  plotting functions. 

Again: The script is set up to expect specific filename structures, and
expects folders to be in a specific arrangement (as they are on 
my [Jessie's] computer). Because of this, it won't work  as a 
self-contained script, but the functions defined in it could be useful,
either if it is imported as a module in another pythons script, or simply 
for reference to see how we've computed the various anomaly features. 

----------------------
DATA PROVIDED
........................................
_stats directories:

The various directories whose names and in 'stats' contain anomaly feature
measurements obtained from our 100,000 synfast simulations, and 1000 SMICA
separated FFP8.1 maps provided on the Planck Legacy Archive.

In the directory names:
- lcdm means synfast simulations (described in paper)
- lcdm+DQ are the same set of synfast simulations, with a dopper quadrupole
   added according to Table 3 of Copi et al arXiv:1311.4562
- smicaffp means the SMICA separated FFP8.1 sims, downgraded to NSIDE=64
- fullsky means that the analysis is done with pseudo-cl measurements of the full sky
- UT78 means that the UT78 mask was used, downgraded to NSIDE=64

In each directory, the data for each stat appear in text files, which have
the realization number in the first column, and the stat measurement in the
second column. They are split into several smaller files (due to how I handled
multiprocessing when creating them); the numbers in each filename show the
range of realizaiton numbers in each file is in the filename. 

For memory considerations, the full cl and c(theta) measurements are not
included, nor are the actual fits files for the
maps, but these could  could be shared upon request.

........................................
COVMATS directory

This directory contains the feature covariances measured from the data in the
stats directories. In the covmats directory there are a number of files.

- The files with the tags -0mean-nounits-unnorm are the covariance matrices.
    > 0mean means the mean over realizations for each quantity is subtracted off
      before computing covmat
    > nounits means the data is divided by the standard deviation over
      realizations for each quantity before computing covmat
    > unnorm just means that the numpy.cov function is used to compute the
      covariance matrix, rather than numpy.corrcoef. (Though, since the data are
      preprocessed by subtracting the mean and dividing by the standard
      deviation before computing the covariances, the covariance and correlation
      coefficients will actually be identical.)
    > the suffix 'lcdm' means they were computed using our fiducial synfast sims
    > the suffix 'ffp' means they were computed using the FFP8.1 sims
    > the suffix 'lcdm+DQ' means they were computed with the synfast sims with
       a doppler quadrupole contribution added

- covmat-samplevar_lcdm_N01000-r00000-99999.dat - contains the sample variance
    of the covariance measured from 1000 sample realizations of the synfast
    (lcdm) sims. The 100 subsampled covmats used to compute this are in the
    directory lcdmcov_N1000_subsample. (Versions with +DQ mean Doppler
    quadrupole has been added)

- covmat-eigendata_lcdm/ffp.dat - contains eigenvalues of corresponding
   covariance matrix in first row, eigen vectors (so PCs) below that. 

----------------------
Feel free to contact me if you have questions about the data or code in this directory!

Script maintained by Jessie Muir (jlmuir@umich.edu or jlynnmuir@gmail.com).
Docstring last updated June 6, 2018.

