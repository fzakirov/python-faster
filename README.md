# python-faster
My adaptation of FASTER bad channel rejection algorithm for EEG data, designed to be used with MNE,
but is not dependent on MNE at all.

Nolan H, Whelan R, Reilly RB. FASTER: Fully Automated Statistical Thresholding for EEG artifact Rejection.
J Neurosci Methods. 2010 Sep 30;192(1):152-62. doi: 10.1016/j.jneumeth.2010.07.015. Epub 2010 Jul 21. PMID: 20654646.

There are already several FASTER implementation in Python exist and some of them much more elaborated than this one,
however, this code exactly replicates Matlab computations from original FASTER (which I found other implementations don't).
