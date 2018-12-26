import numpy as np
from numpy.ctypeslib import ndpointer
from ctypes import c_double, c_bool, c_size_t, c_int, Structure, POINTER

from RUBRIC.nanonet import get_shared_lib

nanonetfilters = get_shared_lib('nanonetfilters')

def compute_sum_sumsq(data):
    """Computer the cumulative sum and cumulative
    sum of squares across a vector

    :param data: the input data

    :returns: the tuple (sums, sumssq) of 1D :class:`numpy.array`s

    :example:

        >>> filters.compute_sum_sumsq([1.0, 2.0, 3.0])
        >>> (array([ 1.,  3.,  6.]), array([  1.,   5.,  14.]))
 
    """
    f = nanonetfilters.compute_sum_sumsq
    f.restype = None
    f.argtypes = [
        ndpointer(dtype='f8', flags='CONTIGUOUS'),
        ndpointer(dtype='f8', flags='CONTIGUOUS'),
        ndpointer(dtype='f8', flags='CONTIGUOUS'),
        c_size_t
    ]
    length = len(data)
    sums = np.empty_like(data)
    sumsqs = np.empty_like(data)
    f(np.ascontiguousarray(data), sums, sumsqs, length)
    return sums, sumsqs


def compute_mave(sums, w_len):
    """Compute the moving average of a vector given
    cumulative sums of the vector

    :param sums: cumulative sum of initial data
    :param w_len: window length of moving average

    :returns: 1D :class:`numpy.array` moving average
    """
    f = nanonetfilters.compute_mave
    f.restype = None
    f.argtypes = [
        ndpointer(dtype='f8', flags='CONTIGUOUS'),
        ndpointer(dtype='f8', flags='CONTIGUOUS'),
        c_size_t,
        c_size_t
    ]
    length = len(sums)
    mave = np.empty_like(sums)
    f(sums, mave, length, w_len)
    return mave


def compute_tstat(sums, sumsqs, w_len, pooled_var = False):
    """Compute a vector of sliding window t-statistics across
    a vector. Each statistic in the output corresponds to left
    and right populations of the input of length `w_len`

    :param sums: cumulative sums across original data
    :param sumsqs: cumulative sum of squares across original
         data
    :param pooled_var: pool variance of left and right populations

    :returns: 1D :class:`numpy.array` of t-statistics
    """
    f = nanonetfilters.compute_tstat
    f.restype = None
    f.argtypes = [
        ndpointer(dtype='f8', flags='CONTIGUOUS'),
        ndpointer(dtype='f8', flags='CONTIGUOUS'),
        ndpointer(dtype='f8', flags='CONTIGUOUS'),
        c_size_t,
        c_size_t,
        c_bool
    ]
    length = len(sums)
    tstat = np.zeros_like(sums)
    f(sums, sumsqs, tstat, length, w_len, pooled_var)
    return tstat


def compute_deltamean(sums, sumsqs, w_len):
    """Compute a vector of sliding window delta-mean across
    a vector. Each statistic in the output corresponds to left
    and right populations of the input of length `w_len`

    :param sums: cumulative sums across original data
    :param sumsqs: cumulative sum of squares across original
         data

    :returns: 1D :class:`numpy.array` of t-statistics
    """
    f = nanonetfilters.compute_deltamean
    f.restype = None
    f.argtypes = [
        ndpointer(dtype='f8', flags='CONTIGUOUS'),
        ndpointer(dtype='f8', flags='CONTIGUOUS'),
        ndpointer(dtype='f8', flags='CONTIGUOUS'),
        c_size_t,
        c_size_t,
    ]
    length = len(sums)
    deltamean = np.zeros_like(sums)
    f(sums, sumsqs, deltamean, length, w_len)
    return deltamean


class Detector(Structure):
    _fields_ = [
        ('DEF_PEAK_POS', c_int),
        ('DEF_PEAK_VAL', c_double),
        ('_signal', POINTER(c_double)),
        ('signal_length', c_size_t),
        ('threshold', c_double),
        ('window_length', c_size_t),
        ('masked_to', c_size_t),
        ('peak_pos', c_int),
        ('peak_value', c_double),
        ('valid_peak', c_bool),
    ]

    def __init__(self, signal, threshold, window_length):
        super(Detector, self).__init__()
        self.DEF_PEAK_POS = -1
        self.DEF_PEAK_VAL = 1e100

        self.signal = np.ascontiguousarray(signal, dtype=c_double)
        self._signal = self.signal.ctypes.data_as(POINTER(c_double))
        self.signal_length = len(signal)
        self.threshold = threshold
        self.window_length = window_length
        self.masked_to = 0
        self.peak_pos = self.DEF_PEAK_POS
        self.peak_value = self.DEF_PEAK_VAL
        self.valid_peak = False


def _construct_events(sums, sumsqs, edges, sample_rate):
    """Creates event data from sums and sumsqs of raw data
    and a list of boundaries

    :param sums: Cumulative sum of raw data points
    :param sumsqs: Cumulative sum of squares of raw data points

    :returns: a 1D :class:`numpy.array` containing event data


    .. note::
       edges is internally trimmed such that all indices are in the
       open interval (0,len(sums)). Events are then constructed as

       [0, edges[0]), [edges[0], edges[1]), ..., [edges[n], len(sums))

    """
    assert len(sums) == len(sumsqs), "sums and sumsqs must be the same length, got {} and  {}".format(len(sums), len(sumsqs))
    assert np.all(edges >= 0), "all edge indices must be positive"
    assert np.all( map( lambda x: isinstance( x, int ), edges ) )
    # We could check maximal values here too, but we're going to get rid of them below.
    #    We put some faith in the caller to have read the DocString above.

    edges = [i for i in edges if i>0 and i<len(sums)]
    edges.append(len(sums))

    num_events = len(edges)
    if sample_rate is not None:
        events = np.empty(num_events, dtype=[('start', float), ('length', float),
                                             ('mean', float), ('stdv', float)])
    else:
        events = np.empty(num_events, dtype=[('start', int), ('length', int),
                                             ('mean', float), ('stdv', float)])

    s = 0
    sm = 0
    smsq = 0
    for i, e in enumerate(edges):
        events['start'][i] = s
        ev_sample_len = e - s
        events['length'][i] = ev_sample_len

        ev_mean = float(sums[e-1] - sm) / ev_sample_len
        events['mean'][i] = ev_mean
        variance = max(0.0, (sumsqs[e-1] - smsq) / ev_sample_len - (ev_mean**2))
        events['stdv'][i] = np.sqrt(variance)
        s = e
        sm = sums[e-1]
        smsq = sumsqs[e-1]

    if sample_rate is not None:
        events['start'] /= sample_rate
        events['length'] /= sample_rate

    return events


def short_long_peak_detector(t_stats, thresholds, window_lengths, peak_height):
    """Peak detector for two signals. Signal derived from shorter window
    length takes precedence.

    Transcribed from find_events_single_scale in wavelet_util.cpp.
    Equivalent to the MinKNOW implementation as of 25/06/2014.

    :param t_stats: Length 2 list of t-statistic signals
    :param thresholds: Length 2 list of thresholds on t-statistics
    :param window_lengths: Length 2 list of window lengths across
        raw data from which `t_stats` are derived
    :peak_height: Absolute height a peak in signal must rise below
        previous and following minima to be considered relevant

    This should be transcribed back to C for speed.

    .. note::
        The peak_height parameter here is rather odd.
    """


    # Create objects analogous to original code, and sort according to length
    detectors = [
        Detector(x[0], x[1], x[2]) for x in zip (t_stats, thresholds, window_lengths)
    ]
    if detectors[0].window_length > detectors[1].window_length:
        detectors = detectors[::-1]

    f = nanonetfilters.short_long_peak_detector
    f.restype = None
    f.argtypes = [POINTER(Detector), POINTER(Detector), c_double, ndpointer(dtype=c_size_t, flags='CONTIGUOUS')]
    peaks = np.zeros(len(detectors[0].signal), dtype=c_size_t)
    f(detectors[0], detectors[1], peak_height, peaks)

    peaks = peaks[np.nonzero(peaks)]
    return peaks.astype(int)


def deltamean_tstat_event_detect(raw_data, sample_rate, window_lengths=[16, 40], thresholds=[8.0, 4.0], peak_height = 1.0):
    """
    Deltamean event detection on short window, tstat event detection with long.

    :param raw_data: ADC values
    :param sample_rate: Sampling rate of data in Hz
    :param window_lengths: Length 2 list of window lengths across
        raw data from which `deltameans` are derived
    :param thresholds: Length 2 list of thresholds on deltameans
    :peak_height: Absolute height a peak in signal must rise below
        previous and following minima to be considered relevant
    """
    sums, sumsqs = compute_sum_sumsq(raw_data)

    short_window = window_lengths[0]
    long_window = window_lengths[1]

    short_window_deltamean = compute_deltamean(sums, sumsqs, short_window)
    long_window_tstat = compute_tstat(sums, sumsqs, long_window)

    stats = [short_window_deltamean,
             long_window_tstat]

    peaks = short_long_peak_detector(stats, thresholds, window_lengths, peak_height)
    events = _construct_events(sums, sumsqs, peaks, sample_rate)

    return events


def deltamean_event_detect(raw_data, sample_rate, window_lengths=[16, 40], thresholds=[8.0, 4.0], peak_height = 1.0):
    """
    Basic, standard even detection comparing mean of two windows

    :param raw_data: ADC values
    :param sample_rate: Sampling rate of data in Hz
    :param window_lengths: Length 2 list of window lengths across
        raw data from which `deltameans` are derived
    :param thresholds: Length 2 list of thresholds on deltameans
    :peak_height: Absolute height a peak in signal must rise below
        previous and following minima to be considered relevant
    """
    sums, sumsqs = compute_sum_sumsq(raw_data)

    deltameans = []
    for i, w_len in enumerate(window_lengths):
        deltamean = compute_deltamean(sums, sumsqs, w_len)
        deltameans.append(deltamean)

    peaks = short_long_peak_detector(deltameans, thresholds, window_lengths, peak_height)
    events = _construct_events(sums, sumsqs, peaks, sample_rate)

    return events


def minknow_event_detect(raw_data, sample_rate, window_lengths=[16, 40], thresholds=[8.0, 4.0], peak_height = 1.0):
    """Basic, standard even detection using two t-tests

    :param raw_data: ADC values
    :param sample_rate: Sampling rate of data in Hz
    :param window_lengths: Length 2 list of window lengths across
        raw data from which `t_stats` are derived
    :param thresholds: Length 2 list of thresholds on t-statistics
    :peak_height: Absolute height a peak in signal must rise below
        previous and following minima to be considered relevant
    """
    sums, sumsqs = compute_sum_sumsq(raw_data)

    tstats = []
    for i, w_len in enumerate(window_lengths):
        tstat = compute_tstat(sums, sumsqs, w_len, False)
        tstats.append(tstat)

    peaks = short_long_peak_detector(tstats, thresholds, window_lengths, peak_height)
    events = _construct_events(sums, sumsqs, peaks, sample_rate)

    return events


