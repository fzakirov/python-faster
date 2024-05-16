import numpy as np
import mne

"""
My adaptation of FASTER bad channel rejection algorithm for EEG data, designed to be used with MNE,
but is not dependent on MNE at all.

Nolan H, Whelan R, Reilly RB. FASTER: Fully Automated Statistical Thresholding for EEG artifact Rejection.
J Neurosci Methods. 2010 Sep 30;192(1):152-62. doi: 10.1016/j.jneumeth.2010.07.015. Epub 2010 Jul 21. PMID: 20654646.

There are already several FASTER implementation in Python exist and some of them much more elaborated than this one,
however, this code exactly replicates Matlab computations from original FASTER (which I found other implementations don't).
"""

def distancematrix(EEG, eeg_chans):
    """
    Compute distance matrices based on EEG channel locations.

    Parameters:
    -----------
    EEG : typically an MNE object
        The EEG object containing channel locations.
    eeg_chans : list
        List of EEG channel indices.

    Returns:
    --------
    distmatrixpol : numpy array
        Polar distance matrix.
    distmatrixxyz : numpy array
        Cartesian distance matrix.
    distmatrixproj : numpy array
        Projected distance matrix.
    """
    
    # Assuming EEG.chanlocs is a list of dicts with keys 'X', 'Y', 'Z', 'theta', 'radius'
    # and EEG.data is a NumPy array

    # Polar distance matrix
    distmatrixpol = np.zeros((len(eeg_chans), len(eeg_chans)))
    for i, chan2tst in enumerate(eeg_chans):
        for j, q in enumerate(eeg_chans):
            theta_diff = np.radians(EEG.chanlocs[i][0]['theta'] - EEG.chanlocs[j][0]['theta'])
            radius1 = EEG.chanlocs[i][0]['radius']
            radius2 = EEG.chanlocs[j][0]['radius']
            distmatrixpol[i, j] = np.sqrt(radius1**2 + radius2**2 - 2 * radius1 * radius2 * np.cos(theta_diff))

    # XYZ coordinates
    Xs = np.array([ch[0]['X'] for ch in EEG.chanlocs])
    Ys = np.array([ch[0]['Y'] for ch in EEG.chanlocs])
    Zs = np.array([ch[0]['Z'] for ch in EEG.chanlocs])

    # Cartesian (XYZ) distance matrix
    distmatrixxyz = np.zeros((len(eeg_chans), len(eeg_chans)))
    for i, u in enumerate(eeg_chans):
        for j, v in enumerate(eeg_chans):
            distmatrixxyz[i, j] = np.sqrt((Xs[i] - Xs[j])**2 + (Ys[i] - Ys[j])**2 + (Zs[i] - Zs[j])**2)

    # Projected distance matrix
    D = np.max(distmatrixxyz)
    distmatrixproj = (np.pi - 2 * np.arccos(distmatrixxyz / D)) * (D / 2)

    return distmatrixpol, distmatrixxyz, distmatrixproj

def min_z(list_properties, rejection_options=None):
    """
    Apply rejection based on z-scores.

    Parameters:
    -----------
    list_properties : numpy array
        Properties of EEG channels.
    rejection_options : dict, optional
        Options for rejection.

    Returns:
    --------
    lengths : numpy array
        Boolean array indicating rows where any specified measure exceeds the threshold.
    """
    
    if rejection_options is None:
        rejection_options = {
            'measure': np.ones(list_properties.shape[1]),
            'z': 3 * np.ones(list_properties.shape[1])
        }
    
    rejection_options['measure'] = np.array(rejection_options['measure'], dtype=bool).astype(int)

    # Calculate z-scores
    zs = list_properties - np.mean(list_properties, axis=0)
    zs = zs / np.std(zs, axis=0, ddof=1)
    zs = np.nan_to_num(zs)

    # Determine which elements exceed the z-score threshold
    all_l = np.abs(zs) > np.tile(rejection_options['z'], (list_properties.shape[0], 1))
    all_l = all_l.astype(int)
    # Return boolean array indicating rows where any specified measure exceeds the threshold
    lengths = np.any(all_l[:, rejection_options['measure']], axis=1)
    return lengths
    
def hurst(x):
    """
    Compute the Hurst exponent of a time series.

    Parameters:
    -----------
    x : numpy array
        Time series data.

    Returns:
    --------
    hurst_exp : float
        Hurst exponent of the time series.
    """
    
    x0 = x.copy()
    x0_len = len(x)
    
    yvals = np.zeros(x0_len)
    xvals = np.zeros(x0_len)
    x1 = np.zeros(x0_len)
    
    index = 0
    binsize = 1
    
    while x0_len > 4:
    
        y = round(np.std(x0, ddof=1), 4)
        index += 1
        xvals[index - 1] = binsize  # Adjusted for zero-based indexing
        yvals[index - 1] = binsize * y
    
        x0_len = x0_len // 2  # Integer division
        binsize *= 2
        for ipoints in range(1, x0_len+1):  # xrange changed to range
            x1[ipoints - 1] = (x0[2 * ipoints - 1] + x0[2 * ipoints - 2]) * 0.5
    
        x0 = x1[:x0_len]
    
    # First value is always 0
    xvals = xvals[:index]
    yvals = yvals[:index]
    
    logx = np.log(xvals)
    logy = np.log(yvals)
    
    p2 = np.polyfit(logx, logy, 1)
    return np.round(p2[0],4)


def channel_properties(EEG, eeg_chans, ref_chan):
    """
    Compute properties of EEG channels.

    Parameters:
    -----------
    EEG : typically an MNE object
        The EEG object containing EEG data.
    eeg_chans : list
        List of EEG channel indices to consider.
    ref_chan : int
        Reference channel index.

    Returns:
    --------
    list_properties : numpy array
        Properties of EEG channels.
    """

    ref_chan = [ref_chan]

# Convert to NumPy array if raw is an MNE object
    if isinstance(EEG, mne.io.eeglab.eeglab.RawEEGLAB):
        data = EEG.data
    
    measure = 0
    
    if ref_chan is not None and len(ref_chan) == 1:
        pol_dist = distancematrix(EEG, eeg_chans)[0]
        dist_inds = np.argsort(pol_dist[ref_chan, eeg_chans])
        idist_inds = np.argsort(dist_inds)
    
    # 1 Mean correlation between each channel and all other channels
    ignore = []
    datacorr = data
    for u in eeg_chans:
        if np.max(data[u, :]) == 0 and np.min(data[u, :]) == 0:
            ignore.append(u)
    
    # Calculate correlations
    calc_indices = set(eeg_chans) - set(ignore)
    ignore_indices = set(eeg_chans).intersection(ignore)
    corrs = np.abs(np.corrcoef(data[list(calc_indices), :]))
    mcorrs = np.zeros(len(eeg_chans))
    for u in range(len(calc_indices)):
        mcorrs[list(calc_indices)[u]] = np.mean(corrs[u, :])
    mcorrs[list(ignore_indices)] = np.mean(mcorrs[list(calc_indices)])
    
    num_measures = 3
    list_properties = np.zeros((len(eeg_chans), num_measures))
    
    # 2 Quadratic correction for distance from reference electrode
    if ref_chan is not None and len(ref_chan) == 1:
        p = np.polyfit(pol_dist[ref_chan, list(dist_inds)], mcorrs[list(dist_inds)], 2)
        fitcurve = np.polyval(p, pol_dist[ref_chan, list(dist_inds)])
        corrected = mcorrs - fitcurve[idist_inds]
        list_properties[:, measure] = corrected
    else:
        list_properties[:, measure] = mcorrs[list(dist_inds)]
    measure += 1
    
    # 3 Variance of the channels
    vars = np.var(data[eeg_chans, :], axis=1)
    vars[~np.isfinite(vars)] = np.mean(vars[np.isfinite(vars)])
    
    # Quadratic correction for distance from reference electrode
    if ref_chan is not None and len(ref_chan) == 1:
        p = np.polyfit(pol_dist[ref_chan, list(dist_inds)], vars[list(dist_inds)], 2)
        fitcurve = np.polyval(p, pol_dist[ref_chan, list(dist_inds)])
        corrected = vars - fitcurve[idist_inds]
        list_properties[:, measure] = corrected
    else:
        list_properties[:, measure] = vars
    measure += 1
    
    # 4 Hurst exponent
    for u in range(len(eeg_chans)):
        list_properties[u, measure] = hurst(data[eeg_chans[u], :])
    
    for u in range(list_properties.shape[1]):
        list_properties[np.isnan(list_properties[:, u]), u] = round(np.nanmean(list_properties[:, u]), 4)
        list_properties[:, u] -= round(np.median(list_properties[:, u]), 4)
    return(list_properties)