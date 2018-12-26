import numpy as np
from math import log
import pkg_resources

from RUBRIC.nanonet import all_kmers, kmers_to_annotated_sequence
from RUBRIC.nanonet import Aligner


def _load_substitution_matrix(fname):
    """ Loads an unwrapped representation of a substitution matrix.
    
    :param fname: Filename of substitution matrix file.
    
    :returns: Representation of matrix in log-space times 100. Values are
        32 bit integers, which should all be negative. State ordering is as
        given by the compute_kmer_mapping() function. Probabilities smaller
        than 6e-6 are given the value of -1200.
    """
    subdata = np.genfromtxt(fname, names=True, dtype=None)
    pos_to_kmer, kmer_to_pos = all_kmers(length=3, rev_map=True)
    matrix = np.empty((64, 64), dtype=np.int32)
    for row in subdata:
        i = kmer_to_pos[row['kmer1']]
        j = kmer_to_pos[row['kmer2']]
        if row['prob'] > 6e-6:
            val = int(log(row['prob']) * 100)
        else:
            val = -1200
        matrix[i, j] = val
    return matrix

matrix_file = pkg_resources.resource_filename('nanonet', 'data/rtc_mismatch_scores.txt')
sub_matrix = _load_substitution_matrix(matrix_file)

open_gap = 500
extend_gap = 500
outside_gap = min(open_gap - 200, extend_gap)
gap_pens = {
    'open0': open_gap,
    'open1': open_gap,
    'start0': outside_gap,
    'start1': outside_gap,
    'end0': outside_gap,
    'end1': outside_gap,
    'extend0': extend_gap,
    'extend1': extend_gap
}


def align_3mer_sequences(sequence0, sequence1, substitution_matrix=sub_matrix, gap_penalties=gap_pens, reverse=True, lowmem=True):
    """Align two sequences in base-space using 3mers.

    :param sequence0: String representing a sequence of bases.
    :param sequence1: String representing a sequence of bases.
    :param substitution_matrix: 64x64 matrix of substitution scores to use for alignment. Should be
        a 2D numpy array of type int32.
    :param gap_penalties: Dictionary of gap penalties. See below.
    :param reverse: Bool indicating whether the second sequence should be reversed.
    :param lowmem: Bool indicating whether to use the (slower) low memory implementation.

    :returns: A tuple of:

        * Numpy record array with fields 'pos0' and 'pos1', representing the alignment.
        * Tuple of a scalar value indicating the alignment score and the average
          continuous alignment length.
    :rtype: tuple

    The gap penalty dictionary should be laid out as follows:
        {start0: penalty for aligning sequence 0 before the start of sequence 1,
         end0: penatly for aligning sequence 0 after the end of sequence 1,
         open0: penalty for aligning sequence 0 to a new gap in sequence 1,
         extend0: penalty for extending a gap within sequence 1,
         start1: penalty for aligning sequence 1 before the start of sequence 2,
         end1: penatly for aligning sequence 1 after the end of sequence 2,
         open1: penalty for aligning sequence 1 to a new gap in sequence 2,
         extend1: penalty for extending a gap within sequence 2,
        }
        The only required field is open0. Gap extension values will default to being the same
        as opening a gap. Start and end gap penalties default to being the same as the extension
        penalty. And the second set of penalties will default to the values for the first.
    .. note::
       Resulting alignment is in terms of 3mers. So '0' represents bases 0-2, and the
       largest  value in the alignment will be len(sequence) - 3. Since the alignment is
       done in terms of 3mers, if the sequence was generated from 5mers then the first and
       last base should be discarded before calling this function.

    """
    submat = [[int(val) for val in line] for line in substitution_matrix]
    pos_to_kmer, kmer_to_pos = all_kmers(length=3, rev_map=True)
    seq0 = [kmer_to_pos[sequence0[i:i+3]] for i in xrange(len(sequence0) - 2)]
    seq1 = [kmer_to_pos[sequence1[i:i+3]] for i in xrange(len(sequence1) - 2)]
    if reverse:
        seq1[:] = seq1[::-1]
    gaps = _gap_penalties_dict_to_list(gap_penalties)
    aligner = Aligner(submat, gaps, lowmem)
    alignment, score = aligner.align(seq0, seq1)
    if reverse:
        for pos in xrange(len(alignment)):
            if alignment[pos][1] != -1:
                alignment[pos] = (alignment[pos][0], len(seq1) - alignment[pos][1] - 1)

    # We'll return the average continuously-aligned length as well.
    alignment_lengths = []
    current_alignment_length = 0
    for pos in alignment:
        if pos[0] == -1 or pos[1] == -1:  # I.e. a stay or skip
            if current_alignment_length > 0:
                alignment_lengths.append(current_alignment_length)
                current_alignment_length = 0
        else:
            current_alignment_length += 1
    if len(alignment_lengths) > 0:
        average_continuous_length = np.average(alignment_lengths)
    else:
        average_continuous_length = current_alignment_length
    npalignment = np.empty(len(alignment), dtype=[('pos0', int), ('pos1', int)])
    npalignment[:] = alignment
    return npalignment, (score, average_continuous_length)


def _gap_penalties_dict_to_list(gap_penalties):
    """ Convert dictionary of gap penalties into an array

    :param gap_penalties: Dictionary of gap penalties

    :returns: List of gap penalties in order which align_1mer_sequences and align_3mer_sequences can use
    """
    gaps = [0] * 8
    gaps[2] = gap_penalties['open0']
    gaps[3] = gap_penalties.get('extend0', gaps[2])
    gaps[0] = gap_penalties.get('start0', gaps[3])
    gaps[1] = gap_penalties.get('end0', gaps[3])
    gaps[6] = gap_penalties.get('open1', gaps[2])
    gaps[7] = gap_penalties.get('extend1', gaps[6])
    gaps[4] = gap_penalties.get('start1', gaps[7])
    gaps[5] = gap_penalties.get('end1', gaps[7])
    return gaps


def align_basecalls(kmers0, kmers1, substitution_matrix=sub_matrix, gap_penalties=gap_pens, lowmem=True):
    """ Align template to complement basecalls, using the align_3mer_sequences function.

    :param kmers0: Template basecalled kmers.
    :param kmers1: Complement basecalled kmers.
    :param substitution_matrix: 64x64 matrix of substitution scores to use for alignment. Should be
        a 2D numpy array of type int32.
    :param gap_penalties: Dictionary of gap penalties. See below.
    :param lowmem: Bool indicating whether to use the (slower) low memory implementation.

    :returns: A tuple of:

        * Numpy array with fields 'pos0' and 'pos1'
        * Scalar value indicating the alignment score and the average continuous
          alignment length.
    :rtype: tuple

    Returns a "filled-in" alignment. So there will be no -1 values (gaps). Instead,
    values can be repeated in either of the sequences.

    The returned alignment is trimmed, meaning that it will start and end with events
    that are aligned to each other. Therefore events at the beginning and end of
    either sequence  may have been removed.

    .. warning:
       It is possible for the alignment to fail, if too few points end up directly aligned
       to each other. In this case the function will return the tuple (None, None).
    """
    sequence0, index0 = kmers_to_annotated_sequence(kmers0)
    sequence1, index1 = kmers_to_annotated_sequence(kmers1)
    kmer_len = len(kmers0[0])
    trim = kmer_len - 3
    trim_left = int((trim + 1) / 2)
    trim_right = int(trim / 2)
    sequence0 = sequence0[trim_left:(len(sequence0) - trim_right)]
    sequence1 = sequence1[trim_left:(len(sequence1) - trim_right)]
    alignment, score = align_3mer_sequences(sequence0, sequence1, substitution_matrix, gap_penalties, reverse=True)
    # Find positions in the alignment that don't have gaps.
    hits = []
    for i, j in alignment:
        if i != -1 and j != -1:
            p0 = index0[i]
            p1 = index1[j]
            if p0 != -1 and p1 != -1:
                hits.append((p0, p1))
    if len(hits) < 2:
        # Not enough aligned positions to do anything sensible.
        return None, None
    # Build up a filled-in alignment by interpolating between aligned positions.
    new_alignment = [hits[0]]
    for i in xrange(1, len(hits)):
        delta0 = hits[i][0] - hits[i-1][0]
        delta1 = hits[i-1][1] - hits[i][1]
        if delta0 > 1 and delta1 > 1:
            # Both sequences jump by more than 1 between aligned points.
            # One sequence should increment by one each position. The other will vary.
            n = max(delta0, delta1) - 1
            p0 = hits[i-1][0]
            p1 = hits[i-1][1]
            step0 = float(delta0 - 1) / float(n)
            step1 = float(delta1 - 1) / float(n)
            for k in range(n):
                p0 += step0
                p1 -= step1
                new_alignment.append((int(round(p0)), int(round(p1))))
        elif delta0 > 1:
            # Need to insert repetions into sequence 2.
            for j in range(hits[i-1][0] + 1, hits[i][0]):
                new_alignment.append((j, hits[i-1][1]))
        elif delta1 > 1:
            # Need to insert repetions into sequence 1.
            for j in range(hits[i-1][1] - 1, hits[i][1], -1):
                new_alignment.append((hits[i-1][0], j))
        else:
            new_alignment.append(hits[i])
    alignment = np.empty(len(new_alignment), dtype=[('pos0', int), ('pos1', int)])
    for n, p in enumerate(new_alignment):
        alignment[n] = p
    return alignment, score
