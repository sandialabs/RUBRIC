from itertools import izip

import numpy as np
import numpy.lib.recfunctions as nprf

from RUBRIC.nanonet import Fast5
from RUBRIC.nanonet import segment
from RUBRIC.nanonet import all_nmers


def padded_offset_array(array, pos):
    """Offset an array and pad with zeros.
    :param array: the array to offset.
    :param pos: offset size, positive values correspond to shifting the
       original array to the left (and padding the end of the output).
    """
    out = np.empty(len(array))
    if pos == 0:
        out = array
    elif pos > 0:
        out[:-pos] = array[pos:]
        out[-pos:] = 0.0
    else:
        out[:-pos] = 0.0
        out[-pos:] = array[0:pos]
    return out


def scale_array(X, with_mean=True, with_std=True, copy=True):
    """Standardize an array
    Center to the mean and component wise scale to unit variance.
    :param X: the data to center and scale.
    :param with_mean: center the data before scaling.
    :param with_std: scale the data to unit variance.
    :param copy: copy data (or perform in-place)
    """    
    X = np.asarray(X)
    if copy:
        X = X.copy()
    if with_mean:
        mean_ = np.mean(X)
        X -= mean_
        mean_1 = X.mean()
        if not np.allclose(mean_1, 0.0):
            X -= mean_1
    if with_std:
        scale_ = np.std(X)
        if scale_ == 0.0:
            scale_ = 1.0
        X /= scale_
        if with_mean:
            mean_2 = X.mean()
            if not np.allclose(mean_2, 0.0):
                X -= mean_2
    return X


def events_to_features(events, window=[-1, 0, 1], sloika_model=False):
    """Read events from a .fast5 and return feature vectors.

    :param filename: path of file to read.
    :param window: list specifying event offset positions from which to
        derive features. A short centered window is used by default.
    """
    
    fg = SquiggleFeatureGenerator(events, sloika_model=sloika_model)
    for pos in window:
        fg.add_mean_pos(pos)
        fg.add_sd_pos(pos)
        fg.add_dwell_pos(pos)
        fg.add_mean_diff_pos(pos)
    X = fg.to_numpy()

    return X


def make_basecall_input_multi(eventList, section='template', window=[-1, 0, 1], trim=10, min_len=1000, max_len=9000,
    event_detect=True, ed_params={'window_lengths':[3, 6], 'thresholds':[1.4, 1.1], 'peak_height':0.2}, sloika_model=True):
    """Like the above, but doesn't yields directly events. The point here is to
    be fully consistent with the currennt interface but allow use of the python
    library
    """
    for eventArr in eventList:
        events, _ = segment(eventArr, section=section) 
        
        try:
            
            X = events_to_features(events, window=window, sloika_model=sloika_model)
            
        except TypeError:
            continue
        try:
            X = X[trim:-trim]
            events = events[trim:-trim]
        except:
            continue
        
        yield  X


def chunker(array, chunk_size):
    """Yield non-overlapping chunks of input.

    :param array: list-like input
    :param chunk_size: output chunk size
    """
    for i in xrange(0, len(array), chunk_size):
        yield array[i:i+chunk_size]


def get_events_ont_mapping(filename, kmer_len=3, section='template'):
    """Scrape event-alignment data from .fast5
    
    :param filename: input file.
    :param section: template or complement
    """
    with Fast5(filename) as fh:
        events, _ = fh.get_any_mapping_data(section=section)
    return events


def get_labels_ont_mapping(filename, kmer_len=3, section='template'):
    """Scrape kmer labels from .fast5 file.

    :param filename: input file.
    :param kmer_len: length of kmers to return as labels.
    :param section: template or complement
    """
    bad_kmer = 'X'*kmer_len
    with Fast5(filename) as fh:
        # just get template mapping data
        events, _ = fh.get_any_mapping_data(section=section)
        base_kmer_len = len(events['kmer'][0])
        if base_kmer_len < kmer_len:
            raise ValueError(
                'kmers in mapping file are {}mers, but requested {}mers.'.format(
                base_kmer_len, kmer_len
            ))
        if base_kmer_len == kmer_len:
            y = events['kmer']
        else:
            k1 = base_kmer_len/2 - kmer_len/2 - 1
            k2 = k1 + kmer_len
            y = np.fromiter(
                (k[k1:k2] for k in events['kmer']),
                dtype='>S{}'.format(kmer_len),
                count = len(events)
            )
        y[~events['good_emission']] = bad_kmer
    return y


def make_currennt_training_input_multi(fast5_files, netcdf_file, window=[-1, 0, 1], kmer_len=3, alphabet='ACGT', chunk_size=1000, min_chunk=900, trim=10, get_events=get_events_ont_mapping, get_labels=get_labels_ont_mapping, callback_kwargs={'section':'template', 'kmer_len':3}):
    """Write NetCDF file for training/validation input to currennt.

    :param fast5_list: list of .fast5 files to process
    :param netcdf_file: output .netcdf file 
    :param window: event window to derive features
    :param kmer_len: length of kmers to learn
    :param alphabet: alphabet of kmers
    :param chunk_size: chunk size to break reads into for SGE batching
    :param min_chunk: minimum chunk size (used to discard remainder of reads
    :param trim: no. of feature vectors to trim (from either end)
    :param get_events: callback to return event data, will be passed .fast5 filename
    :param get_labels: callback to return event kmer labels, will be passed .fast5 filename
    :param callback_kwargs: kwargs for both `get_events` and `get_labels`
    """
    from netCDF4 import Dataset

    # We need to know ahead of time how wide our feature vector is,
    #    lets generate one and take a peek. Check also callbacks
    #    produce meaningful data.
    X = events_to_features(get_events(fast5_files[0], **callback_kwargs), window=window)
    labels = get_labels(fast5_files[0], **callback_kwargs)
    inputPattSize = X.shape[1]
    if len(X) != len(labels):
        raise RuntimeError('Length of features and labels not equal.')
    
    # Our state labels are kmers plus a junk kmer
    kmers = all_nmers(kmer_len, alpha=alphabet)
    bad_kmer = 'X'*kmer_len
    kmers.append(bad_kmer)
    all_kmers = {k:i for i,k in enumerate(kmers)}

    with Dataset(netcdf_file, "w", format="NETCDF4") as ncroot:
        #Set dimensions
        ncroot.createDimension('numSeqs', None)
        ncroot.createDimension('numLabels', len(all_kmers))
        ncroot.createDimension('maxSeqTagLength', 10)
        ncroot.createDimension('numTimesteps', None)
        ncroot.createDimension('inputPattSize', inputPattSize)

        #Set variables
        seqTags = ncroot.createVariable("seqTags", 'S1', ("numSeqs", "maxSeqTagLength"))
        seqLengths = ncroot.createVariable("seqLengths", 'i4', ("numSeqs",))
        inputs = ncroot.createVariable("inputs", 'f4', ("numTimesteps", "inputPattSize"))
        targetClasses = ncroot.createVariable("targetClasses", 'i4', ("numTimesteps",))

        chunks_written = 0
        for i, f in enumerate(fast5_files):
            try: # lot of stuff
                # Run callbacks to get features and labels
                X = events_to_features(get_events(f, **callback_kwargs), window=window)
                labels = get_labels(f, **callback_kwargs)
            except:
                print "Skipping: {}".format(f)
                continue

            try:   
                X = X[trim:-trim]
                labels = labels[trim:-trim]
                if len(X) != len(labels):
                    raise RuntimeError('Length of features and labels not equal.')
            except:
                print "Skipping: {}".format(f)

            try:
                # convert kmers to ints
                y = np.fromiter(
                    (all_kmers[k] for k in labels),
                    dtype=np.int16, count=len(labels)
                )
            except Exception as e:
                # Checks for erroneous alphabet or kmer length
                raise RuntimeError(
                    'Could not convert kmer labels to ints in file {}. '
                    'Check labels are no longer than {} and contain only {}'.format(f, kmer_len, alphabet)
                )
            else:
                print "Adding: {}".format(f)
                for chunk, (X_chunk, y_chunk) in enumerate(izip(chunker(X, chunk_size), chunker(y, chunk_size))):
                    if len(X_chunk) < min_chunk:
                        break
                    chunks_written += 1 #should be the same as curr_numSeqs below

                    seqname = "S{}_{}".format(i, chunk)
                    _seqTags = np.zeros(10, dtype="S1")
                    _seqTags[:len(seqname)] = list(seqname)

                    numTimesteps = len(X_chunk)
                    curr_numSeqs = len(ncroot.dimensions["numSeqs"])
                    curr_numTimesteps = len(ncroot.dimensions["numTimesteps"])

                    seqTags[curr_numSeqs] = _seqTags
                    seqLengths[curr_numSeqs] = numTimesteps
                    inputs[curr_numTimesteps:] = X_chunk
                    targetClasses[curr_numTimesteps:] = y_chunk

    return chunks_written, inputPattSize, kmers


class SquiggleFeatureGenerator(object):
    def __init__(self, events, labels=None, sloika_model=False):
        """Feature vector generation from events.

        :param events: standard event array.
        :param labels: labels for events, only required for training

        ..note:
            The order in which the feature adding methods is called should be the
            same for both training and basecalling.
        """
        self.events = np.copy(events)
        self.labels = labels
        self.features = {}
        self.feature_order = []

        # Augment events
        if sloika_model:
            delta = np.abs(np.ediff1d(events['mean'], to_end=0))
            self.events = nprf.append_fields(events, 'delta', delta)
            for field in ('mean', 'stdv', 'length', 'delta'):
                scale_array(self.events[field], copy=False)
        else:
            for field in ('mean', 'stdv', 'length',):
                scale_array(self.events[field], copy=False)
            delta = np.ediff1d(self.events['mean'], to_begin=0)
            scale_array(delta, with_mean=False, copy = False)
            self.events = nprf.append_fields(self.events, 'delta', delta)
 
    def to_numpy(self):
        out = np.empty((len(self.events), len(self.feature_order)))
        for j, key in enumerate(self.feature_order):
            out[:, j] = self.features[key]
        return out

    def _add_field_pos(self, field, pos):
        tag = "{}[{}]".format(field, pos)
        if tag in self.features:
            return self
        self.feature_order.append(tag)
        self.features[tag] = padded_offset_array(self.events[field], pos)
        return self

    def add_mean_pos(self, pos):
        self._add_field_pos('mean', pos)
        return self

    def add_sd_pos(self, pos):
        self._add_field_pos('stdv', pos)
        return self

    def add_dwell_pos(self, pos):
        self._add_field_pos('length', pos)
        return self

    def add_mean_diff_pos(self, pos):
        self._add_field_pos('delta', pos)
        return self

