#!/usr/bin/env python
import argparse
import itertools
import os
import sys
import timeit
import warnings
from functools import partial

import numpy as np
import pkg_resources

from features import make_basecall_input_multi
from RUBRIC.nanonet import decoding, nn, cl
from RUBRIC.nanonet.cmdargs import FileExist, CheckCPU, AutoBool
from RUBRIC.nanonet import Fast5, iterate_fast5, short_names
from RUBRIC.nanonet import JobQueue
from RUBRIC.nanonet import FastaWrite, tang_imap, kmer_overlap, group_by_list, AddFields, nanonet_resource

warnings.simplefilter("ignore")

now = timeit.default_timer

__fast5_analysis_name__ = 'Basecall_RNN_1D'
__fast5_section_name__ = 'BaseCalled_{}'
__ETA__ = nn.tiny

__DEFAULTS__ = {
    'r9.4': {
        'ed_params': {
            'window_lengths':[3, 6], 'thresholds':[1.4, 1.1],
            'peak_height':0.2
        },
        'model': nanonet_resource('r9.4_template.npy'),
        'sloika_model': True,
    },
    'r9': {
        'ed_params': {
            'window_lengths':[5, 10], 'thresholds':[2.0, 1.1],
            'peak_height':1.2
        },
        'model': nanonet_resource('r9_template.npy'),
        'template_model': nanonet_resource('r9_template.npy'),
        'complement_model': nanonet_resource('r9_complement.npy'),
        'sloika_model': False,
    }
}
__DEFAULT_CHEMISTRY__ = 'r9.4'


class SetChemistryDefaults(argparse.Action):
    """Check if the input file exist."""

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, values)
        params = __DEFAULTS__[values]
        for key, value in params.items():
            setattr(namespace, key, value)


class ParseEventDetect(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, {
            'window_lengths': values[0:2], 'thresholds': values[2:5],
            'peak_height': values[5]
        })


def get_parser():
    parser = argparse.ArgumentParser(
        description="""A 1D RNN basecaller for Oxford Nanopore data.

This software is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.

(c) 2016 Oxford Nanopore Technologies Ltd.
""",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("input", action=FileExist, nargs='?', #--list_platforms means this can be absent
        help="A path to fast5 files.")
    parser.add_argument("--watch", default=None, type=int,
        help="Switch to watching folder, argument value used as timeout period.")
    parser.add_argument("--section", default=None, choices=('template', 'complement'),
        help="Section of read for which to produce basecalls, will override that stored in model file.")

    parser.add_argument("--chemistry", choices=__DEFAULTS__.keys(), default=None, action=SetChemistryDefaults,
        help="Shorthand for selection various analysis parameters.")
    parser.add_argument("--model", type=str, action=FileExist,
        default=pkg_resources.resource_filename('nanonet', 'data/default_template.npy'),
        help="Trained RNN.")
    parser.add_argument("--sloika_model", action='store_true', default=False,
        help="Use sloika style feature normalization.")
    parser.add_argument("--event_detect", default=True, action=AutoBool,
        help="Perform event detection, else use existing event data.")
    parser.add_argument("--ed_params", default=__DEFAULTS__[__DEFAULT_CHEMISTRY__]['ed_params'],
        metavar=('window0', 'window1', 'threshold0', 'threshold1', 'peakheight'), nargs=5,
        action=ParseEventDetect,
        help="Event detection parameters")

    parser.add_argument("--output", type=str,
        help="Output name, output will be in fasta format.")
    parser.add_argument("--fastq", action=AutoBool, default=False,
        help="Output fastq rather than fasta.")
    parser.add_argument("--write_events", action=AutoBool, default=False,
        help="Write event datasets to .fast5.")
    parser.add_argument("--strand_list", default=None, action=FileExist,
        help="List of reads to process.")
    parser.add_argument("--limit", default=None, type=int,
        help="Limit the number of input for processing.")
    parser.add_argument("--min_len", default=500, type=int,
        help="Min. read length (events) to basecall.")
    parser.add_argument("--max_len", default=50000, type=int,
        help="Max. read length (events) to basecall.")

    parser.add_argument("--jobs", default=1, type=int, action=CheckCPU,
        help="No of decoding jobs to run in parallel.")

    parser.add_argument("--trans", type=float, nargs=3, default=None,
        metavar=('stay', 'step', 'skip'), help='Base transition probabilities')
    parser.add_argument("--fast_decode", action=AutoBool, default=False,
        help="Use simple, fast decoder with no transition estimates.")

    parser.add_argument("--exc_opencl", action=AutoBool, default=False,
        help="Do not use CPU alongside OpenCL, overrides --jobs.")
    parser.add_argument("--list_platforms", action=AutoBool, default=False,
        help="Output list of available OpenCL GPU platforms.")
    parser.add_argument("--platforms", nargs="+", type=str, default=None,
        help="List of OpenCL GPU platforms and devices to be used in a format VENDOR:DEVICE:N_Files space separated, i.e. --platforms nvidia:0:1 amd:0:2 amd:1:2.")

    return parser

class ProcessAttr(object):
    def __init__(self, use_opencl=False, vendor=None, device_id=0):
        self.use_opencl = use_opencl
        self.vendor = vendor
        self.device_id = device_id

def list_opencl_platforms():
    if cl is None:
        raise ImportError('pyopencl is not installed, install with pip.')
    print '\n' + '=' * 60 + '\nOpenCL Platforms and Devices'
    platforms = (p for p in cl.get_platforms() if p.get_devices(device_type=cl.device_type.GPU))
    for platform in platforms:
        print '=' * 60
        print 'Name:     {}'.format(platform.name)
        print 'Vendor:   {}'.format(platform.vendor)
        print 'Version:  {}'.format(platform.version)
        for device in platform.get_devices(device_type=cl.device_type.GPU):
            print '    ' + '-' * 56
            print '    Name:  {}'.format(device.name)
            print '    Type:  {}'.format(cl.device_type.to_string(device.type))
            print '    Max Clock Speed:  {} Mhz'.format(device.max_clock_frequency)
            print '    Compute Units:  {}'.format(device.max_compute_units)
            print '    Local Memory:  {:.0f} KB'.format(device.local_mem_size/1024)
            print '    Constant Memory:  {:.0f} KB'.format(device.max_constant_buffer_size/1024)
            print '    Global Memory: {:.0f} GB'.format(device.global_mem_size/1073741824.0)
    print


def process_read(events, network, min_prob=1e-5, trans=None, for_2d=False, write_events=True, fast_decode=False, **kwargs):
    """Run neural network over a set of fast5 files

    :param modelfile: neural network specification.
    :param fast5: read file to process
    :param post_only: return only the posterior matrix
    :param **kwargs: kwargs of make_basecall_input_multi
    """
    #sys.stderr.write("CPU process\n processing {}\n".format(fast5))
    #print 'we made it into NNC process read!!'
    network = network
    kwargs['window'] = network.meta['window']
    #print network
    
    # Get features
    
    it = make_basecall_input_multi((events,), **kwargs)
    features = it.next()
    #print features
        
        
    #print 'show me the money'
    # Run network
    t0 = now()
    post = network.run(features.astype(nn.dtype))
    #print post
    #print 'raga' + str(len(post))
    # Manipulate posterior matrix
    post, good_events = clean_post(post, network.meta['kmers'], min_prob)

    # Decode kmers
    t0 = now()
    if fast_decode:
        score, states = decoding.decode_homogenous(post, log=False)
    else:
        trans = decoding.fast_estimate_transitions(post, trans=trans)
        score, states = decoding.decode_profile(post, trans=np.log(__ETA__ + trans), log=False)
    decode_time = now() - t0

    # Form basecall
    kmers = [x for x in network.meta['kmers'] if 'X' not in x]
    qdata = get_qdata(post, kmers)
    seq, qual, kmer_path = form_basecall(qdata, kmers, states, qscore_correction=None)


#    rtn_value = [(fname, (seq, qual), score, len(features)), (network_time, decode_time)]
#    if for_2d:
#        trans = np.sum(trans, axis=0)
#        trans /= np.sum(trans)
#        rtn_value.append((post, kmer_path, trans, kmers))
    #print str(seq)
	#added a return qual and length
    return seq, qual, len(seq)


def clean_post(post, kmers, min_prob):
    # Do we have an XXX kmer? Strip out events where XXX most likely,
    #    and XXX states entirely
    if kmers[-1] == 'X'*len(kmers[-1]):
        bad_kmer = post.shape[1] - 1
        max_call = np.argmax(post, axis=1)
        good_events = (max_call != bad_kmer)
        post = post[good_events]
        post = post[:, :-1]
        if len(post) == 0:
            return None, None
    
    weights = np.sum(post, axis=1).reshape((-1,1))
    post /= weights 
    post = min_prob + (1.0 - min_prob) * post
    return post, good_events


def get_qdata(post, kmers):
    bases = sorted(set(''.join(kmers)))
    kmer_len = len(kmers[0])
    n_events = len(post)
    n_bases = len(bases)

    qdata = np.empty((n_events, len(bases)*kmer_len), dtype=post.dtype)
    for i, (pos, base) in enumerate(itertools.product(range(kmer_len), bases)):
        cols = np.fromiter((k[pos] == base for k in kmers),
            dtype=bool, count=len(kmers))
        qdata[:, i] = np.sum(post[:, cols], axis=1)
    return qdata


def form_basecall(qdata, kmers, states, qscore_correction=None):
    bases = sorted(set(''.join(kmers)))
    kmer_len = len(kmers[0])
    n_events = len(qdata)
    n_bases = len(bases)
    kmer_path = [kmers[i] for i in states]

    moves = kmer_overlap(kmer_path)
    seq_len = np.sum(moves) + kmer_len
    scores = np.zeros((seq_len, len(bases)), dtype=np.float32)
    sequence = list(kmer_path[0])
    posmap = range(kmer_len)

    _contribute(scores, qdata[0, :], posmap, n_bases)
    for event, move in enumerate(moves):
        if move > 0:
            if move == kmer_len:
                posmap = range(posmap[-1] + 1, posmap[-1] + 1 + kmer_len)
            else:
                posmap[:-move] = posmap[move:]
                posmap[-move:] = range(posmap[-move-1] + 1, posmap[-move-1] + 1 + move)
            sequence.append(kmer_path[event][-move:])
        _contribute(scores, qdata[event, :], posmap, n_bases)
    sequence = ''.join(sequence)
    base_to_pos = {b:i for i,b in enumerate(bases)}
    scores += __ETA__
    scoresums = np.sum(scores, axis=1)
    scores /= scoresums[:, None]
    called_probs = np.fromiter(
        (scores[n, base_to_pos[base]] for n, base in enumerate(sequence)),
        dtype=float, count=len(sequence)
    )

    if qscore_correction == 'template':
        # initial scores fit to empirically observed probabilities
        #   per score using: Perror = a.10^(-bQ/10). There's a change
        #   in behaviour at Q10 so we fit two lines. (Q10 seems suspicious).
        switch_q = 10
        a, b = 0.05524, 0.70268
        c, d = 0.20938, 1.00776
        switch_p = 1.0 - np.power(10.0, - 0.1 * switch_q)
        scores = np.empty_like(called_probs)
        for x, y, indices in zip((a,c), (b,d), (called_probs < switch_p, called_probs >= switch_p)):
            scores[indices] = -(10.0 / np.log(10.0)) * (y*np.log1p(-called_probs[indices]) + np.log(x))
    elif qscore_correction in ('2d','complement'):
        # same fitting as above
        if qscore_correction == 'complement':
            x, y = 0.13120, 0.88952
        else:
            x, y = 0.02657, 0.65590 
        scores = -(10.0 / np.log(10.0)) * (y*np.log1p(-called_probs) + np.log(x))
    else:
        scores = -10.0 * np.log1p(-called_probs) / np.log(10.0)

    offset = 33
    scores = np.clip(np.rint(scores, scores).astype(int) + offset, offset, 126)
    qstring = ''.join(chr(x) for x in scores)
    return sequence, qstring, kmer_path


def _contribute(scores, qdata, posmap, n_bases):
     kmerlen = len(posmap)
     for kmer_pos, seq_pos in enumerate(posmap):
         index = (kmerlen - kmer_pos - 1) * n_bases
         scores[seq_pos, :] += qdata[index:index + n_bases]
     return


def write_to_file(fast5, events, section, seq, qual, good_events, kmer_path, kmers, post, states):
    adder = AddFields(events[good_events])
    adder.add('model_state', kmer_path,
        dtype='>S{}'.format(len(kmers[0])))
    adder.add('p_model_state', np.fromiter(
        (post[i,j] for i,j in itertools.izip(xrange(len(post)), states)),
        dtype=float, count=len(post)))
    adder.add('mp_model_state', np.fromiter(
        (kmers[i] for i in np.argmax(post, axis=1)),
        dtype='>S{}'.format(len(kmers[0])), count=len(post)))
    adder.add('p_mp_model_state', np.max(post, axis=1))
    adder.add('move', np.array(kmer_overlap(kmer_path)), dtype=int)

    mid = len(kmers[0]) / 2
    bases = set(''.join(kmers)) - set('X')
    for base in bases:
        cols = np.fromiter((k[mid] == base for k in kmers),
            dtype=bool, count=len(kmers))
        adder.add('p_{}'.format(base), np.sum(post[:, cols], axis=1), dtype=float)

    events = adder.finalize()

    with Fast5(fast5, 'a') as fh:
       base = fh._join_path(
           fh.get_analysis_new(__fast5_analysis_name__),
           __fast5_section_name__.format(section))
       fh._add_event_table(events, fh._join_path(base, 'Events'))
       try:
           name = fh.get_read(group=True).attrs['read_id']
       except:
           name = fh.filename_short
       fh._add_string_dataset(
           '@{}\n{}\n+\n{}\n'.format(name, seq, qual),
           fh._join_path(base, 'Fastq'))

        
def process_read_opencl(modelfile, eventList, min_prob=1e-5, trans=None, write_events=True, fast_decode=False, **kwargs):
    """Run neural network over a set of fast5 files

    :param modelfile: neural network specification.
    :param fast5: read file to process
    :param post_only: return only the posterior matrix
    :param **kwargs: kwargs of make_basecall_input_multi
    """
    #sys.stderr.write("OpenCL process\n processing {}\n{}\n".format(fast5_list, pa.__dict__))

    network = np.load(modelfile).item()
    kwargs['window'] = network.meta['window']

    # Get features

    features_list =  make_basecall_input_multi(eventList,**kwargs)


    features_list = [x.astype(nn.dtype) for x in features_list] 

#    features_list = features_list.next()
    #print features_list
    # Set up OpenCL
    devices = cl.get_platforms()[0].get_devices() 
    device = devices[0]
    
    max_workgroup_size = device.get_info(cl.device_info.MAX_WORK_GROUP_SIZE)
    ctx = cl.Context([device])
    queue_list = [cl.CommandQueue(ctx)] * len(features_list)
#    print features_list[0][0]
    # Run network

    post_list = network.run(features_list, ctx, queue_list)
    

    # Manipulate posterior
    post_list, good_events_list = zip(*(
        clean_post(post, network.meta['kmers'], min_prob) for post in post_list
    ))

    # Decode kmers

    if fast_decode:
        # actually this is slower, but we want to run the same algorithm
        #   in the case of heterogeneous computer resource.
        score_list, states_list = zip(*(
            decoding.decode_homogenous(post, log=False) for post in post_list
        ))
    else:
        trans_list = [np.log(__ETA__ +
                             decoding.fast_estimate_transitions(post, trans=trans))
            for post in post_list]
        score_list, states_list = decoding.decode_profile_opencl(
            ctx, queue_list, post_list, trans_list=trans_list,
            log=False, max_workgroup_size=max_workgroup_size
        )
    
            
    # Form basecall
    kmers = network.meta['kmers']
    seq_list = []
    qual_list = []
    kmer_path_list = []
    for states, post in zip(states_list, post_list):
        seq, qual, kmer_path = form_basecall(post, [x for x in network.meta['kmers'] if 'X' not in x], states)
        seq_list.append(seq)
        kmer_path_list.append(kmer_path)
        qual_list.append(qual)

    # Write events table
#    if write_events:
#        section_list = (kwargs['section'] for _ in xrange(n_files))
#        kmers_list = (network.meta['kmers'] for _ in xrange(n_files))
#        for data in zip(
#            file_list, events_list, section_list, seq_list, qual_list,
#            good_events_list, kmer_path_list, kmers_list, post_list, states_list
#            ):
#            write_to_file(*data)

    # Construst a sequences of objects as process_read returns
    data = zip(seq_list, qual_list), score_list, (len(x) for x in features_list)

    ret = data
    
    return ret


def main():
    if len(sys.argv) == 1:
        sys.argv.append("-h")
    args = get_parser().parse_args()
 
    if args.list_platforms:
        list_opencl_platforms() 
        sys.exit(0)
        
    modelfile  = os.path.abspath('data/r9.4_template.npy')

    #TODO: handle case where there are pre-existing files.
    if args.watch is not None:
        # An optional component
        from RUBRIC.nanonet import Fast5Watcher
        initial_jobs = iterate_fast5(args.input, paths=True) 
        fast5_files = Fast5Watcher(args.input, timeout=args.watch, initial_jobs=initial_jobs)
    else:
        sort_by_size = 'desc' if args.platforms is not None else None
        fast5_files = iterate_fast5(args.input, paths=True, strand_list=args.strand_list, limit=args.limit, sort_by_size=sort_by_size)

    fix_args = [
        modelfile
    ]
    fix_kwargs = {a: getattr(args, a) for a in ( 
        'min_len', 'max_len', 'section',
        'event_detect', 'fast_decode',
        'write_events', 'ed_params', 'sloika_model'
    )}

    # Define worker functions   
    workers = []
    if not args.exc_opencl:
        cpu_function = partial(process_read, *fix_args, **fix_kwargs)
        workers.extend([(cpu_function, None)] * args.jobs)
    if args.platforms is not None:
        if cl is None:
            raise ImportError('pyopencl is not installed, install with pip.')
        for platform in args.platforms:
            vendor, device_id, n_files = platform.split(':')
            pa = ProcessAttr(use_opencl=True, vendor=vendor, device_id=int(device_id))
            fargs = fix_args + [pa]
            opencl_function = partial(process_read_opencl, *fargs, **fix_kwargs)
            workers.append(
                (opencl_function, int(n_files))
            )

    # Select how to spread load
    if args.platforms is None:
        # just CPU
        worker, n_files = workers[0]
        mapper = tang_imap(worker, fast5_files, threads=args.jobs, unordered=True)
    elif len(workers) == 1:
        # single opencl device
        #    need to wrap files in lists, and unwrap results
        worker, n_files = workers[0]
        fast5_files = group_by_list(fast5_files, [n_files])
        mapper = itertools.chain.from_iterable(itertools.imap(worker, fast5_files))
    else:
        # Heterogeneous compute
        mapper = JobQueue(fast5_files, workers)

    # Off we go
    n_reads = 0
    n_bases = 0
    n_events = 0
    timings = [0.0, 0.0]
    t0 = now()
    with FastaWrite(args.output, args.fastq) as fasta:
        for result in mapper:
            if result is None:
                continue
            data, time = result
            fname, call_data, _, n_ev = data
            name, _ = short_names(fname)
            basecall, quality = call_data
            if args.fastq:
                fasta.write(name, basecall, quality)
            else:
                fasta.write(name, basecall)
            n_reads += 1
            n_bases += len(basecall)
            n_events += n_ev
            timings = [x + y for x, y in zip(timings, time)]
    t1 = now()
    sys.stderr.write('Basecalled {} reads ({} bases, {} events) in {}s (wall time)\n'.format(n_reads, n_bases, n_events, t1 - t0))
    if n_reads > 0:
        network, decoding  = timings
        sys.stderr.write(
            'Run network: {:6.2f} ({:6.3f} kb/s, {:6.3f} kev/s)\n'
            'Decoding:    {:6.2f} ({:6.3f} kb/s, {:6.3f} kev/s)\n'
            .format(
                network, n_bases/1000.0/network, n_events/1000.0/network,
                decoding, n_bases/1000.0/decoding, n_events/1000.0/decoding,
            )
        )


def run_basecall(events, network):
    return process_read(events, network)
