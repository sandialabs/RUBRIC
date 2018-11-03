""" Module for performing 2D basecalls.

Within this module a generic form is used for representing alignments
of two sequences to each other. This is done with a strictured array,
with columns named 'pos0' and 'pos1'. These take the following form:

pos0 = [1, 2, -1, 3, 4, 5, 6]
pos1 = [1, 2,  3, 4, 5, 6, 7]

indicating that the 1st event of the first sequence aligns to the 1st
in the second, -1 indicating the corresponding event in the other
sequences does not have a partner.
"""

import os
import numpy as np
from math import ceil, floor
from RUBRIC.nanonet import viterbi_2d
from RUBRIC.nanonet import rc_kmer, kmers_to_sequence
from RUBRIC.nanonet import align_basecalls
from RUBRIC.nanonet import get_qdata, form_basecall

try:
    from RUBRIC.nanonet import viterbi_2d_ocl
except ImportError:
    viterbi_2d_ocl = None


def init_opencl_device(cpu_id=0):
    if viterbi_2d_ocl is None:
        return None
    proxy_cl = viterbi_2d_ocl.proxyCL()

    vendors, error = proxy_cl.available_vendors()
    if error or not vendors:
        print "Error establishes OpenCL vendors"
        return None

    # Initially choose the first vendor from the list.
    # If there is more than one opencl vendor available then give the priority to nvida gpu
    # then amd which can be either gpu or cpu.
    active_vendor = vendors[0] 
    if len(vendors) > 1:
        opencl_vendor = viterbi_2d_ocl.vendor
        if opencl_vendor.nvidia in vendors:
            active_vendor = opencl_vendor.nvidia
            proxy_cl.enable_cuda_build_cache(False) # we don't want to build the same kernel in PROD over and over again.
        elif opencl_vendor.amd in vendors:
            active_vendor = opencl_vendor.amd
        elif opencl_vendor.intel in vendors:
            active_vendor = opencl_vendor.intel
        elif opencl_vendor.apple in vendors:
            active_vendor = opencl_vendor.apple
    ret, error = proxy_cl.select_vendor(active_vendor)
    if not ret or error:
         print "Error selecting OpenCL vendor"
         return None
    opencl_device_type = viterbi_2d_ocl.device_type

    devices, error = proxy_cl.available_devices()
    if error or not devices:
        print "Error establishes OpenCL devices"
        return None

    if len(devices) == 1:
        ret, error = proxy_cl.select_device(devices[0].id)
        if ret and not error:
            #print "Selected OpenCL device:", devices[0].type, devices[0].name # TODO: remove or log to log file.
            ret, error = proxy_cl.create_context()
            if not ret or error:
                print "Error creating context for device"
                return None
            return proxy_cl
    else:
        # Give priority to gpu, then cpu, then whatever else is available.
        dev_lst = [device_info.id for device_info in devices if device_info.type == opencl_device_type.gpu]
        if dev_lst:
            device_to_use = cpu_id % len(dev_lst)
            ret, error = proxy_cl.select_device(dev_lst[device_to_use])
            if ret and not error:
                #print "Selected OpenCL device", cpu_id, device_to_use, devices[device_to_use].type, devices[device_to_use].name
                # TODO: remove or log to log file.
                ret, error = proxy_cl.create_context()
                if not ret or error:
                    print "Error creating context for device"
                    return None
                return proxy_cl

        dev_lst = [device_info.id for device_info in devices if device_info.type == opencl_device_type.cpu]
        if dev_lst:
            ret, erro = proxy_cl.select_device(dev_lst[0])
            if ret and not error:
                #print "Selected OpenCL device:", devices[0].type, devices[0].name # TODO: remove or log to log file
                ret, error = proxy_cl.create_context()
                if not ret or error:
                    print "Error creating context for device."
                    return None
                return proxy_cl
        ret, error = proxy_cl.select_device(devices[0].id)
        if ret and not error:
            #print "Selected OpenCL device:", devices[0].type, devices[0].name # TODO: remove or log to log file
            ret, error = proxy_cl.create_context()
            if not ret or error:
                print "Error creating context for device."
                return None
            return proxy_cl

    return None


def reverse_complement_posteriors(post, kmers):
    """ Transform posteriors to a reverse-complement representation.
    
    :param post: A 2d :class:`numpy.ndarray` (events x states).
    :param kmers: list identifying kmers
    :returns: A new array, with the posteriors in reverse-complement order. 
    """
    results = np.empty(post.shape, post.dtype)
    rc_order = np.argsort([rc_kmer(x) for x in kmers])
    return post[::-1, rc_order]
    

def apply_em_weights(post, weights):
    """ Apply emission event weights to posteriors.
    :param post: A 2d :class:`numpy.ndarray` (events x states).
    :param weights: A 1d :class:`numpy.ndarray` of event emission weights.
    """
    num_states = post.shape[1]
    num_events = post.shape[0]
    for i in range(num_events):
        offset = (1.0 - weights[i]) / num_states
        post[i,:] *= weights[i]
        post[i,:] += offset
    return


class Chunker(object):
    """Support class to handle the chunking of the 2D basecall."""

    def __init__(self, alignment, chunk_size=500):
        self.chunk_size = chunk_size
        a = 0
        b = alignment.size - 1
        while alignment[a]['pos0'] == -1 or alignment[a]['pos1'] == -1:
            a += 1
        while alignment[b]['pos0'] == -1 or alignment[b]['pos1'] == -1:
            b -= 1
        self.template_start = alignment[a]['pos0']
        self.template_end = alignment[b]['pos0']
        self.complement_start = alignment[b]['pos1']
        self.complement_end = alignment[a]['pos1']
        self.trimmed_alignment = alignment[a:b+1].copy()
        for i, align in enumerate(self.trimmed_alignment):
            if align['pos0'] == -1:
                self.trimmed_alignment[i]['pos0'] = self.trimmed_alignment[i-1]['pos0']
            else:
                self.trimmed_alignment[i]['pos0'] -= self.template_start
            if align['pos1'] == -1:
                self.trimmed_alignment[i]['pos1'] = self.trimmed_alignment[i-1]['pos1']
            else:
                self.trimmed_alignment[i]['pos1'] -= self.complement_start
        
        # Set chunk end points.
        num_chunks = int(ceil(float(self.trimmed_alignment.size) / float(self.chunk_size)))
        if num_chunks == 0:
            num_chunks = 1
        points = np.linspace(self.trimmed_alignment.size / num_chunks, self.trimmed_alignment.size,
                         num=num_chunks, endpoint=True)
        chunk_ends = [int(floor(point)) for point in points]
        chunk_ends[-1] = self.trimmed_alignment.size - 1

        # Add 1 to each end point, so that it is a past-the-end position.
        self.chunk_ends = [end + 1 for end in chunk_ends]
        self.chunk_starts = [0] + [end for end in self.chunk_ends[:-1]]
        self.num_chunks = num_chunks
        self.chunk_alignments = []
        self.chunk_template_starts = []
        self.chunk_template_ends = []
        self.chunk_complement_starts = []
        self.chunk_complement_ends = []
        for n in range(num_chunks):
            chunk_align = self.trimmed_alignment[self.chunk_starts[n]:self.chunk_ends[n]].copy()
            tmp_chunk_start, tmp_chunk_end = chunk_align[0]['pos0'], chunk_align[-1]['pos0']
            com_chunk_start, com_chunk_end = chunk_align[-1]['pos1'], chunk_align[0]['pos1']
            for i, align in enumerate(chunk_align):
                chunk_align[i]['pos0'] -= tmp_chunk_start
                chunk_align[i]['pos1'] -= com_chunk_start
            self.chunk_alignments.append(chunk_align)
            self.chunk_template_starts.append(tmp_chunk_start)
            self.chunk_template_ends.append(tmp_chunk_end)
            self.chunk_complement_starts.append(com_chunk_start)
            self.chunk_complement_ends.append(com_chunk_end)
        return
    
    def update(self, n, xpos, ypos):
        tmp_chunk_start = self.chunk_template_starts[n]
        tmp_chunk_end = self.chunk_template_ends[n]
        com_chunk_start = self.chunk_complement_starts[n]
        com_next_chunk_start = self.chunk_complement_starts[n+1]
        xend = self.chunk_alignments[n][-1]['pos0']
        
        # Interpolate prior alignment from new starting position to old chunk end.
        deltax = xend - xpos
        deltay = ypos
        maxdelta = max(deltax, deltay)
        interpolate = np.zeros(maxdelta + 1, dtype=self.chunk_alignments[n].dtype)
        for i in xrange(maxdelta + 1):
            if deltax > deltay:
                xval = xpos + i
                yval = ypos - int(round(float(deltay * i) / maxdelta))
            else:
                yval = ypos - i
                xval = xpos + int(round(float(deltax * i) / maxdelta))
            interpolate[i]['pos0'] = xval - xpos
            interpolate[i]['pos1'] = yval + com_chunk_start - com_next_chunk_start
        new_chunk_alignment = np.zeros(self.chunk_alignments[n+1].size + interpolate.size, dtype=interpolate.dtype)
        delta = self.chunk_template_starts[n+1] - tmp_chunk_end + interpolate[-1]['pos0']
        self.chunk_alignments[n+1]['pos0'] += delta
        new_chunk_alignment[:interpolate.size] = interpolate
        new_chunk_alignment[interpolate.size:] = self.chunk_alignments[n+1]
        self.chunk_alignments[n+1] = new_chunk_alignment
        self.chunk_template_starts[n+1] -= delta
        self.chunk_complement_ends[n+1] = self.chunk_complement_starts[n] + ypos
        return


def check_alignment(len1, len2, out_align):
    max1 = np.max(out_align['pos0'])
    max2 = np.max(out_align['pos1'])
    min1 = np.min(out_align['pos0'])
    min2 = np.min(out_align['pos1'])
    if max1 >= len1 or max2 >= len2 or min1 < -1 or min2 < -1:
        raise Exception('Bad alignment. Range1 = ({}, {}) and Range2 = ({}, {}), but Lenght1 = {} and Length2 = {}.'.format(min1, max1, min2, max2, len1, len2))
    return


def call_aligned_pair(posts, transitions, alignment, allkmers, call_band=15,
                      chunk_size=500, use_opencl=False, cpu_id=0):
    """Take posteriors, transitions and an alignment to produce a chunked multi dimensional
    basecall via a Viterbi algorithm.

    :param posts: tuple of posterior matrices.
    :param tuple transitions: tuple of lists each containings [stay, step, skip] probabilities.
    :param alignment: ndarray with fields pos0 and pos1 as produced by :func:`_make_align`.
    :param allkmers: list of all possible kmers.
    :param call_band: width of band to contrain basecall around the alignment generated.
    :param chunk_size: size of chunks to break up the prior alignment into.
    :param use_opencl: whether or not to use GPU acceleration.
    :param cpu_id: CPU id for GPU opencl support.

    :returns: None if the chunking fails, otherwise a tuple of:

        * string containing the sequence predicted by the 2d basecaller
        * list containing the kmers predicted by the 2d basecaller
        * 2d alignment of the two sets of input events
    :rtype: tuple or None

    """
    # Init opencl
    proxy_cl = None
    if use_opencl:
        proxy_cl = init_opencl_device(cpu_id)
        if not proxy_cl:
            return None
 
    # Prepare models and transitions for the viterbi_2d code.
    trans = transitions[0] + transitions[1]
    num_states = len(allkmers)
    kmerlen = len(allkmers[0])
    state_info = {'kmers': list(allkmers)}
    rc_order = np.argsort([rc_kmer(x) for x in allkmers])

    params = {'band_size': call_band,
              'kmer_len': kmerlen,
              'seq2_is_rc': True,
              'use_sd': True, # Meaningless when using posteriors
              'max_nodes': chunk_size * call_band * 6,
              'max_len': chunk_size * 2,
              'stay1': trans[0],
              'step1': trans[1],
              'skip1': trans[2],
              'stay2': trans[3],
              'step2': trans[4],
              'skip2': trans[5]}

    viterbi = None
    if proxy_cl is None:
        viterbi = viterbi_2d.Viterbi2D(state_info, params)
    else:
        viterbi = viterbi_2d_ocl.Viterbi2Docl(proxy_cl)
        viterbi.init_data(state_info, params)
        # Initialize opencl kernel specifics
        work_group_size = 0 # when set to 0 max device available will be used
        src_kernel_dir = os.path.dirname(viterbi_2d_ocl.__file__)
        bin_kernel_dir = os.path.join(os.path.expanduser('~'), '.nanonet_opencl')
        try:
            os.makedirs(bin_kernel_dir)
        except:
            pass

        enable_fp64 = True # whether to offload double floating point calculations to GPU
        ret, error = viterbi.init_cl(src_kernel_dir, bin_kernel_dir, enable_fp64, num_states, work_group_size)
        if not ret or error:
            enable_fp64 = False
            ret, error = viterbi.init_cl(src_kernel_dir, bin_kernel_dir, enable_fp64, num_states, work_group_size)
            if not ret or error:
                return None

    alignment_overlap = max(int(1.5 * call_band), 20)
    data_overlap = max(call_band, 10)

    full_sequence = ''
    qdata = []
    qscores = []

    chunker = Chunker(alignment, chunk_size)
    num_chunks = chunker.num_chunks
    temp_start = chunker.template_start
    temp_end = chunker.template_end
    comp_start = chunker.complement_start
    comp_end = chunker.complement_end

    # Cycle through chunks.
    kmers = []
    max_align = chunker.trimmed_alignment.size * 3
    prior_alignment = chunker.trimmed_alignment.copy()
    new_alignment = np.zeros(max_align, dtype=[('pos0', int), ('pos1', int)])
    prior = None
    start = 0
    cum_pos = 0
    for chunk in range(num_chunks):
        # Extract the data chunk, and the corresponding alignment.
        chunk_align = chunker.chunk_alignments[chunk]
        tmp_chunk_start = chunker.chunk_template_starts[chunk]
        tmp_chunk_end = chunker.chunk_template_ends[chunk] + 1
        com_chunk_start = chunker.chunk_complement_starts[chunk]
        com_chunk_end = chunker.chunk_complement_ends[chunk] + 1

        post1 = posts[0][tmp_chunk_start+temp_start:tmp_chunk_end+temp_start, :].copy()
        post2 = posts[1][com_chunk_start+comp_start:com_chunk_end+comp_start, :]
        post2 = post2[::-1, rc_order].copy()

        chunk_tmp_size = len(post1)
        chunk_com_size = len(post2)

        # Perform the 2D call for the chunk.
        stay_weight1 = np.ones(chunk_tmp_size, dtype=np.float64, order='C')
        stay_weight2 = np.ones(chunk_com_size, dtype=np.float64, order='C')

        n = chunk_com_size - 1
        align_in = []
        for item in chunk_align:
            x0 = item['pos0']
            x1 = n - item['pos1'] if item['pos1'] != -1 else -1
            align_in.append((x0, x1))

        results = viterbi.call_post(post1, post2, stay_weight1, stay_weight2, align_in, prior)
        chunk_kmers = results['kmers']
        chunk_align_out = results['alignment']

        # Need to check for nonsense calls, indicated by mostly stays in the basecall.
        sequence = kmers_to_sequence(chunk_kmers)
        if len(sequence) < len(align_in) / 3:
            return None

        merged_qdata = make_aligned_qdata(post1, post2, chunk_align_out, allkmers)
        
        for i, item in enumerate(chunk_align_out):
            x0 = item[0]
            x1 = n - item[1] if item[1] != -1 else -1
            chunk_align_out[i] = (x0, x1)

        # Find the start position for the next chunk.
        if chunk == num_chunks - 1:
            pos = len(chunk_align_out)
            search = False
        else:
            pos = len(chunk_align_out) - alignment_overlap
            search = True
        while pos >= 0 and search:
            x, y = chunk_align_out[pos][0], chunk_align_out[pos][1]
            if x != -1 and y != -1:
                if x < chunk_tmp_size - data_overlap and y > data_overlap:
                    search = False
            if search:
                pos -= 1
        if pos < len(chunk_align_out) and pos < chunk_size / 2:
            # Chunking has failed. Return null object to indicate this.
            return None
       
        if chunk != num_chunks - 1:
            temp_event, comp_event = chunk_align_out[pos][0], chunk_align_out[pos][1]
            chunker.update(chunk, temp_event, comp_event)
            prior = chunk_kmers[pos-1]

        for i, align in enumerate(chunk_align_out):
            if align[0] == -1:
                chunk_align_out[i] = (-1, chunk_align_out[i][1] + com_chunk_start)
            elif align[1] == -1:
                chunk_align_out[i] = (chunk_align_out[i][0] + tmp_chunk_start, -1)
            else:
                chunk_align_out[i] = (chunk_align_out[i][0] + tmp_chunk_start, chunk_align_out[i][1] + com_chunk_start)

        # Accumulate results from chunk.
        qdata.append(merged_qdata[:pos,:])
        new_alignment[cum_pos:cum_pos+pos] = chunk_align_out[:pos]
        kmers.extend(chunk_kmers[:pos])
        cum_pos += pos

    out_align = new_alignment[0:cum_pos]
    check_alignment(temp_end - temp_start + 1, comp_end - comp_start + 1, out_align)
    # finalised qscore data and form basecall
    qdata = merge_qdata(qdata)
    kmer_map = {k:i for i,k in enumerate(allkmers)}
    states = [kmer_map[k] for k in kmers[0:cum_pos]]
    out_sequence, out_qstring, out_kmers = form_basecall(
        qdata, allkmers, states, qscore_correction='2d'
    ) 

    # Adjust alignment so indexes refer to full non-trimmed data sets.
    for i, align in enumerate(out_align):
        if align['pos0'] != -1:
            out_align[i]['pos0'] += temp_start
        if align['pos1'] != -1:
            out_align[i]['pos1'] += comp_start
            
    return out_sequence, out_qstring, out_kmers, out_align


def make_aligned_qdata(post1, post2, alignment, kmers):
    post = np.empty((len(alignment), post1.shape[1]), dtype=post1.dtype)
    for pos in range(len(alignment)):
        pos0 = alignment[pos][0]
        pos1 = alignment[pos][1]
        temp = None
        comp = None
        if pos0 != -1:
            temp = post1[pos0,:]
        if pos1 != -1:
            comp = post2[pos1,:]
        if temp is None:
            post[pos,:] = comp
        elif comp is None:
            post[pos,:] = temp
        else:
            prod = temp * comp
            nrm = 1.0 / np.sum(prod)
            post[pos,:] = prod * nrm
    qdata = get_qdata(post, kmers)
    return qdata


def merge_qdata(qdata_list):
    n = 0
    for block in qdata_list:
        n += block.shape[0]
    qdata = np.empty((n, block.shape[1]), dtype=block[0].dtype)
    p = 0
    for block in qdata_list:
        qdata[p:p+block.shape[0],:] = block
        p += block.shape[0]
    return qdata


def call_2d(posts, kmers, transitions, allkmers, call_band=15, chunk_size=500, use_opencl=False, cpu_id=0):
    """Wrapper around borrowed code to perform 2D call from posterior matrices and 1D calls.

    :param posts: posterior matrices for template and complement.
    :param kmers: kmer lists for template and complement.
    :param transitions: transition parameters (stay, step, skip) for template and complement.
    :param allkmers: list of all possible kmers.
    :param call_band: width of band to contrain basecall around the alignment generated.
    :param chunk_size: size of chunks to break up the prior alignment into.
    :param use_opencl: whether or not to use GPU acceleration.
    :param cpu_id: CPU id for GPU opencl support.

    :returns: tuple containing sequence, kmers, event alignment.
    """
    failed_alignment = RuntimeError('Failed to align template and complement basecalls.')
    failed_call = RuntimeError('Failed to produce 2D call.')


    try:
        alignment, score = align_basecalls(*kmers)
    except Exception as e:
        raise failed_alignment
    else:
        if alignment is None:
            raise failed_alignment

    try:
        sequence, out_kmers, out_align, trimmed_align = call_aligned_pair(
            posts, transitions, alignment, allkmers, call_band=call_band,
            chunk_size=chunk_size, use_opencl=use_opencl, cpu_id=cpu_id)
    except Exception as e:
        raise failed_call

    return sequence, out_kmers, out_align, trimmed_align


