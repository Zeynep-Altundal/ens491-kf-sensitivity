# experiments/common

Shared statistical utilities for the experiment packages: percentile bootstrap confidence
intervals over independent Monte Carlo runs, including paired differences between two settings or
filters evaluated on the same runs.

## Bootstrap convention (used by every `closure.py`)

* Unit of resampling: one whole Monte Carlo run, in every bootstrap in this repository (runs are
  independent; epochs within a run are never resampled individually because they are correlated).
* Statistic: the reported metric recomputed on each resample, e.g. the mean over runs of the
  per-run time-averaged NEES, or the RMSE as the square root of the mean over runs of the per-run
  mean squared error (all runs of a setting have the same number of epochs).
* Number of resamples **B = 10 000**; two-sided **95 %** percentile interval; resampling generator
  seeded with **2026** so that intervals are reproducible.
* Paired differences: both settings or filters are resampled with the same run indices, and the
  statistic of the difference (A minus B) is computed on each resample.
* A difference is called **supported** when the 95 % interval of the paired difference excludes
  zero, and **not supported** otherwise. A consistency claim (for example "mean NEES equals its
  expected value") is called supported when the interval of the mean contains the expected value.
