# processing_functions.py
import numpy as np
import multiprocessing
import time
import scipy
from scipy import stats

def read_full_spectra_centroid(spectra_obj, max_val = None):
    r"""Build nominal mass mass spectra from centroided mass and intensity values.

    Parameters
    ----------
    spectra_obj :
        Spectra object wrapping chromato dims, all spectra masses, all spectra intensities, mass range_min and range_max.
    Returns
    -------
    A: tuple
        Return the nominal mass mass spectra, _, _.
    Examples
    --------
    >>> import read_chroma
    >>> import mass_spec
    >>> chromato_obj = read_chroma.read_chroma(filename, mod_time)
    >>> chromato,time_rn,spectra_obj = chromato_obj
    >>> spectra, debuts, fins = mass_spec.read_full_spectra_centroid(spectra_obj=spectra_obj)
    """
    start_time = time.time()
    (l1, l2, mv, iv, range_min, range_max) = spectra_obj
    minima = argrelextrema(mv, np.less)[0]
    fins = minima - 1
    fins = np.append(fins, len(mv)-1)
    debuts = np.insert(minima, 0, 0)
    #Stack 1-D arrays as columns into a 2-D array.
    mass_spectra_ind = np.column_stack((debuts, fins))
    spectra = [(mv[beg:end], iv[beg:end]) for beg, end in mass_spectra_ind]
    spectra_full_nom = []

    '''for i in range(len(spectra)):
        spectra_full_nom.append(centroid_to_full_nominal(spectra_obj, spectra[i][0], spectra[i][1]))'''
    cpu_count = multiprocessing.cpu_count()
    mv = np.linspace(range_min, range_max, range_max - range_min + 1).astype(int)
    with multiprocessing.Pool(processes=cpu_count) as pool:
        results = pool.starmap(centroid_to_full_nominal, 
                            [((range_min, range_max), spectra[i][0], spectra[i][1]) for i in range(len(spectra))])
    spectra_full_nom = [(mv, result) for result in results]
    
    # Pad with zeros to have necessary length of spectra
    required_length = l1 * l2
    if len(spectra_full_nom) < required_length:
        spectra_full_nom.extend([(mv, np.zeros_like(mv))] * (required_length - len(spectra_full_nom)))

    print("--- %s seconds --- to compute full spectra centroid" % (time.time() - start_time))

    spectra_full_nom = np.array(spectra_full_nom)
    return spectra_full_nom, debuts, fins

def full_spectra_to_chromato_cube(full_spectra, spectra_obj, mass_range_min=None, mass_range_max=None):
    r"""Compute 3D chromatogram from mass spectra. Then it is possible to read specific mass spectrum from this 3D chromatogram or detect peaks in 3D.

    Parameters
    ----------
    full_spectra :
        Tuple wrapping spectra, debuts and fins.
    spectra_obj :
        Tuple wrapping l1, l2, mv, iv, range_min and range_max
    -------
    Returns
    -------
    A: ndarray
        Return the created 3D chromatogram. An array containing all mass chromatogram.
    -------
    Examples
    --------
    >>> import read_chroma
    >>> chromato_obj = read_chroma.read_chroma(file, mod_time)
    >>> chromato,time_rn,spectra_obj = chromato_obj
    >>> full_spectra = mass_spec.read_full_spectra_centroid(spectra_obj=spectra_obj)
    >>> chromato_cube = read_chroma.full_spectra_to_chromato_cube(full_spectra=full_spectra, spectra_obj=spectra_obj)
    """
    spectra, debuts, fins = full_spectra
    l1, l2, mv, iv, range_min, range_max = spectra_obj

    if (not mass_range_min):
        mass_range_min = range_min
    if (not mass_range_max):
        mass_range_max = range_max
    chromato_mass_list = []

    for tm in range(mass_range_min - range_min, mass_range_max - range_min + 1):

        chromato_mass = spectra[:,1,tm]
        chromato_mass_tm = np.reshape(chromato_mass[:l1*l2], (l2,l1))
        chromato_mass_list.append(chromato_mass_tm)
    return np.array(chromato_mass_list)

def centroid_to_full_nominal(spectra_obj, mass_values, int_values):
    #(l1, l2, mv, iv, range_min, range_max) = spectra_obj
    range_min, range_max = spectra_obj
    #print(range_min)
    #print(range_max)

    #mv = np.linspace(range_min, range_max, range_max - range_min + 1).astype(int)
    iv = np.zeros((range_max - range_min + 1))
    for i, mass in enumerate(mass_values):
        rounded_mass = round(mass)
        mass_ind = rounded_mass - range_min
        iv[mass_ind] = iv[mass_ind] + int_values[i]
    #return mv, iv
    return iv

def slice_at_axis(sl, axis):
    """
    Construct tuple of slices to slice an array in the given dimension.

    Parameters
    ----------
    sl : slice
        The slice for the given dimension.
    axis : int
        The slice for the given dimension.
        The axis to which `sl` is applied. All other dimensions are left
        "unsliced".

    Returns
    -------
    sl : tuple of slices
        A tuple with slices matching `shape` in length.

    Examples
    --------
    >>> slice_at_axis(slice(None, 3, -1), 1)
    (slice(None, None, None), slice(None, 3, -1), Ellipsis)
    """
    return (slice(None),) * axis + (sl,) + (...,)

def sigma_est_dwt(detail_coeffs, distribution='Gaussian'):
    """Calculate the robust median estimator of the noise standard deviation.

    Parameters
    ----------
    detail_coeffs : ndarray
        The detail coefficients corresponding to the discrete wavelet
        transform of an image.
    distribution : str
        The underlying noise distribution.

    Returns
    -------
    sigma : float
        The estimated noise standard deviation (see section 4.2 of [1]_).

    References
    ----------
    .. [1] D. L. Donoho and I. M. Johnstone. "Ideal spatial adaptation
       by wavelet shrinkage." Biometrika 81.3 (1994): 425-455.
       :DOI:`10.1093/biomet/81.3.425`
    """
    # Consider regions with detail coefficients exactly zero to be masked out
    detail_coeffs = detail_coeffs[np.nonzero(detail_coeffs)]

    if distribution.lower() == 'gaussian':
        # 75th quantile of the underlying, symmetric noise distribution
        denom = scipy.stats.norm.ppf(0.75)
        sigma = np.median(np.abs(detail_coeffs)) / denom
    else:
        raise ValueError("Only Gaussian noise estimation is currently " "supported")
    return sigma

def argrelextrema(data, comparator, axis=0, order=1, mode='clip'):
    """
    Calculate the relative extrema of `data`.

    Parameters
    ----------
    data : ndarray
        Array in which to find the relative extrema.
    comparator : callable
        Function to use to compare two data points.
        Should take two arrays as arguments.
    axis : int, optional
        Axis over which to select from `data`. Default is 0.
    order : int, optional
        How many points on each side to use for the comparison
        to consider ``comparator(n, n+x)`` to be True.
    mode : str, optional
        How the edges of the vector are treated. 'wrap' (wrap around) or
        'clip' (treat overflow as the same as the last (or first) element).
        Default is 'clip'. See `numpy.take`.

    Returns
    -------
    extrema : tuple of ndarrays
        Indices of the maxima in arrays of integers. ``extrema[k]`` is
        the array of indices of axis `k` of `data`. Note that the
        return value is a tuple even when `data` is 1-D.

    See Also
    --------
    argrelmin, argrelmax

    Notes
    -----

    .. versionadded:: 0.11.0

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.signal import argrelextrema
    >>> x = np.array([2, 1, 2, 3, 2, 0, 1, 0])
    >>> argrelextrema(x, np.greater)
    (array([3, 6]),)
    >>> y = np.array([[1, 2, 1, 2],
    ...               [2, 2, 0, 0],
    ...               [5, 3, 4, 4]])
    ...
    >>> argrelextrema(y, np.less, axis=1)
    (array([0, 2]), array([2, 1]))

    """
    results = boolrelextrema(data, comparator,
                              axis, order, mode)
    return np.nonzero(results)

def boolrelextrema(data, comparator, axis=0, order=1, mode='clip'):
    """
    Calculate the relative extrema of `data`.

    Relative extrema are calculated by finding locations where
    ``comparator(data[n], data[n+1:n+order+1])`` is True.

    Parameters
    ----------
    data : ndarray
        Array in which to find the relative extrema.
    comparator : callable
        Function to use to compare two data points.
        Should take two arrays as arguments.
    axis : int, optional
        Axis over which to select from `data`. Default is 0.
    order : int, optional
        How many points on each side to use for the comparison
        to consider ``comparator(n,n+x)`` to be True.
    mode : str, optional
        How the edges of the vector are treated. 'wrap' (wrap around) or
        'clip' (treat overflow as the same as the last (or first) element).
        Default 'clip'. See numpy.take.

    Returns
    -------
    extrema : ndarray
        Boolean array of the same shape as `data` that is True at an extrema,
        False otherwise.

    See also
    --------
    argrelmax, argrelmin

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.signal._peak_finding import _boolrelextrema
    >>> testdata = np.array([1,2,3,2,1])
    >>> _boolrelextrema(testdata, np.greater, axis=0)
    array([False, False,  True, False, False], dtype=bool)

    """
    if (int(order) != order) or (order < 1):
        raise ValueError('Order must be an int >= 1')

    datalen = data.shape[axis]
    locs = np.arange(0, datalen)

    results = np.ones(data.shape, dtype=bool)
    main = data.take(locs, axis=axis, mode=mode)
    for shift in range(1, order + 1):
        plus = data.take(locs + shift, axis=axis, mode=mode)
        minus = data.take(locs - shift, axis=axis, mode=mode)
        results &= comparator(main, plus)
        results &= comparator(main, minus)
        if ~results.any():
            return results
    return results


def read_spectrum_from_chromato_cube(pic_coord, chromato_cube):
    #return spectra[pic_coord[1] * chromato.shape[0] + pic_coord[0]]
    ms_from_chromato_cube = chromato_cube[:, pic_coord[0],pic_coord[1]]
    #return np.linspace(range_min, range_max, range_max - range_min + 1).astype(int), ms_from_chromato_cube
    return ms_from_chromato_cube
