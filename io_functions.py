# io_functions.py
import netCDF4 as nc
import numpy as np
import math
import time
import functools
import scipy
import pybaselines
import multiprocessing
import warnings

from processing_functions import read_full_spectra_centroid
from processing_functions import full_spectra_to_chromato_cube
from processing_functions import slice_at_axis
from scipy.signal import savgol_filter
from pybaselines import whittaker
from pybaselines.whittaker import asls
from processing_functions import sigma_est_dwt

def read_chromato_and_chromato_cube(filename, mod_time=1.25, pre_process=True):
    """
    Opens the CDF file and processes the chromatogram.
    
    Returns:
      chromato, time_rn, chromato_cube, sigma, (range_min, range_max)
    """
    start_time = time.time()
    chromato_obj = read_chroma(filename, mod_time)
    chromato, time_rn, spectra_obj = chromato_obj
    (l1, l2, mv, iv, range_min, range_max) = spectra_obj
    print("chromato read in", time.time()-start_time, "s")
    
    start_time = time.time()
    full_spectra = read_full_spectra_centroid(spectra_obj)
    print("full spectra computed in", time.time()-start_time, "s")
    
    # Compute chromatogram cube from full spectra
    chromato_cube = full_spectra_to_chromato_cube(full_spectra, spectra_obj)
    
    # Baseline correction (if needed)
    if pre_process:
        chromato = chromato_no_baseline(chromato)
        chromato_cube = np.array(chromato_cube_corrected_baseline(chromato_cube))
        print("baseline corrected")
    
    sigma = estimate_sigma(chromato, channel_axis=None)
    return chromato, time_rn, chromato_cube, sigma, (range_min, range_max)

def read_chroma(filename, mod_time=1.25, max_val=None):
    ds = nc.Dataset(filename)
    chromato = ds['total_intensity'][:]
    Timepara = ds["scan_acquisition_time"][np.abs(ds["point_count"]) < np.iinfo(np.int32).max]
    sam_rate = 1 / np.mean(Timepara[1:] - Timepara[:-1])
    l1 = math.floor(sam_rate * mod_time)
    l2 = math.floor(len(chromato) / l1)
    
    if max_val:
        mv = ds["mass_values"][:max_val]
        iv = ds["intensity_values"][:max_val]
    else:
        mv = ds["mass_values"][:]
        iv = ds["intensity_values"][:]
    
    range_min = math.ceil(ds["mass_range_min"][:].min())
    range_max = math.floor(ds["mass_range_max"][:].max())
    
    chromato = np.reshape(chromato[:l1 * l2], (l2, l1))
    return chromato, (ds['scan_acquisition_time'][0] / 60, ds['scan_acquisition_time'][-1] / 60), (l1, l2, mv, iv, range_min, range_max)

def chromato_no_baseline(chromato, j=None):
    r"""Correct baseline and apply savgol filter.
    ----------
    chromato : ndarray
        Input chromato.
    Returns
    -------
    chromato :
        The input chromato without baseline
    Examples
    --------
    >>> import read_chroma
    >>> import baseline_correction
    >>> chromato_obj = read_chroma.read_chroma(filename, mod_time)
    >>> chromato,time_rn,spectra_obj = chromato_obj
    >>> chromato = baseline_correction.chromato_no_baseline(chromato)
    """
    tmp = np.empty_like(chromato)
    for i in range (tmp.shape[1]):
        tmp[:,i] = savgol_filter(chromato[:,i] - pybaselines.whittaker.asls(chromato[:,i], lam=1000.0, p=0.05)[0], 5, 2, mode='nearest')
    tmp[tmp < .0] = 0
    return tmp

def estimate_sigma(image, average_sigmas=False, *, channel_axis=None):
    """
    Robust wavelet-based estimator of the (Gaussian) noise standard deviation.

    Parameters
    ----------
    image : ndarray
        Image for which to estimate the noise standard deviation.
    average_sigmas : bool, optional
        If true, average the channel estimates of `sigma`.  Otherwise return
        a list of sigmas corresponding to each channel.
    channel_axis : int or None, optional
        If ``None``, the image is assumed to be grayscale (single-channel).
        Otherwise, this parameter indicates which axis of the array corresponds
        to channels.

        .. versionadded:: 0.19
           ``channel_axis`` was added in 0.19.

    Returns
    -------
    sigma : float or list
        Estimated noise standard deviation(s).  If `multichannel` is True and
        `average_sigmas` is False, a separate noise estimate for each channel
        is returned.  Otherwise, the average of the individual channel
        estimates is returned.

    Notes
    -----
    This function assumes the noise follows a Gaussian distribution. The
    estimation algorithm is based on the median absolute deviation of the
    wavelet detail coefficients as described in section 4.2 of [1]_.

    References
    ----------
    .. [1] D. L. Donoho and I. M. Johnstone. "Ideal spatial adaptation
       by wavelet shrinkage." Biometrika 81.3 (1994): 425-455.
       :DOI:`10.1093/biomet/81.3.425`

    Examples
    --------
    .. testsetup::
        >>> import pytest; _ = pytest.importorskip('pywt')

    >>> import skimage.data
    >>> from skimage import img_as_float
    >>> img = img_as_float(skimage.data.camera())
    >>> sigma = 0.1
    >>> rng = np.random.default_rng()
    >>> img = img + sigma * rng.standard_normal(img.shape)
    >>> sigma_hat = estimate_sigma(img, channel_axis=None)
    """
    try:
        import pywt
    except ImportError:
        raise ImportError(
            'PyWavelets is not installed. Please ensure it is installed in '
            'order to use this function.'
        )

    if channel_axis is not None:
        channel_axis = channel_axis % image.ndim
        _at = functools.partial(slice_at_axis, axis=channel_axis)
        nchannels = image.shape[channel_axis]
        sigmas = [
            estimate_sigma(image[_at(c)], channel_axis=None) for c in range(nchannels)
        ]
        if average_sigmas:
            sigmas = np.mean(sigmas)
        return sigmas
    elif image.shape[-1] <= 4:
        msg = (
            f'image is size {image.shape[-1]} on the last axis, '
            f'but channel_axis is None. If this is a color image, '
            f'please set channel_axis=-1 for proper noise estimation.'
        )
        warnings(msg)
    coeffs = pywt.dwtn(image, wavelet='db2')
    detail_coeffs = coeffs['d' * image.ndim]
    return sigma_est_dwt(detail_coeffs, distribution='Gaussian')

def chromato_cube_corrected_baseline(chromato_cube):
    r"""Apply baseline correction on each chromato of the input.
    ----------
    chromato_cube :
        Input chromato.
    Returns
    -------
    chromato_cube:
        List of chromato from input list without baseline
    Examples
    --------
    >>> import read_chroma
    >>> import baseline_correction
    >>> chromato_obj = read_chroma.read_chroma(filename, mod_time)
    >>> chromato,time_rn,spectra_obj = chromato_obj
    >>> full_spectra = mass_spec.read_full_spectra_centroid(spectra_obj=spectra_obj)
    >>> chromato_cube = read_chroma.full_spectra_to_chromato_cube(full_spectra=full_spectra, spectra_obj=spectra_obj)
    >>> chromato_cube = np.array(baseline_correction.chromato_cube_corrected_baseline(chromato_cube))
    """
    cpu_count = multiprocessing.cpu_count()
    chromato_cube_no_baseline = []
    with multiprocessing.Pool(processes = cpu_count) as pool:
        for i, result in enumerate(pool.starmap(chromato_no_baseline, [(m_chromato, j) for j, m_chromato in enumerate(chromato_cube)])):
            chromato_cube_no_baseline.append(result)
    return chromato_cube_no_baseline



# Here you could also include any other IO-related helper functions like:



# (or import them if you have already structured them in a different file)