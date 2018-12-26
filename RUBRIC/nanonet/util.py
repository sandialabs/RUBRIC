import imp
import importlib
import math
import os
import random
import string
import sys
from contextlib import contextmanager
from ctypes import cdll
from functools import partial
from itertools import tee, imap, izip, izip_longest, product, cycle, islice, chain, repeat
from multiprocessing import Pool

import numpy as np
import pkg_resources
from numpy.lib import recfunctions as nprf

__eta__ = 1e-100

def get_shared_lib(name):
    """Cross-platform resolution of shared-object libraries, working
    around vagueries of setuptools
    """
    try:
        # after 'python setup.py install' we should be able to do this
        lib_file = importlib.import_module(name).__file__
    except Exception as e:
        print 'FAILED IMPORT'
        print e
        try:
            # after 'python setup.py develop' this should work
            lib_file = imp.find_module(name)[1]
        except Exception as e:
            raise ImportError('Cannot locate C library for event detection.')
        else:
            lib_file = os.path.abspath(lib_file)
    finally:
        # lib_file = r'C:\Users\hedwa\PycharmProjects\RUBRIC\nanonet\decoding.py'
        library = cdll.LoadLibrary(lib_file)
    return library


def nanonet_resource(filename, subfolder='data'):
    return pkg_resources.resource_filename('nanonet',
        os.path.join(subfolder, filename))


comp = {
    'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C', 'X': 'X', 'N': 'N',
    'a': 't', 't': 'a', 'c': 'g', 'g': 'c', 'x': 'x', 'n': 'n',
    '-': '-'
}


def all_kmers(alphabet='ACGT', length=5, rev_map=False):
    """ Find all possible kmers of given length.

    .. Warning::
       The return value type of this function is dependent on the input
       arguments

    :param alphabet: string from which to draw characters
    :param length: length of kmers required
    :param rev_map: return also a dictionary containing the reverse mapping i.e. {'AAA':0, 'AAC':1}

    :returns: a list of strings. kmers are sorted by the ordering of the *alphabet*. If *rev_map*
        is specified a second item is returned as noted above.

    """
    fwd_map = map(lambda x: ''.join(x), product(alphabet, repeat=length))
    if not rev_map:
        return fwd_map
    else:
        return fwd_map, dict(zip(fwd_map, xrange(len(fwd_map))))


def all_nmers(n=3, alpha='ACGT'):
    return all_kmers(length=n, alphabet=alpha)


def com(k):
    """ Return complement of base.

    Performs the subsitutions: A<=>T, C<=>G, X=>X for both upper and lower
    case. The return value is identical to the argument for all other values.
    """
    try:
        return comp[k]
    except KeyError:
        sys.stderr.write("WARNING: No reverse complement for {} found, returning argument.".format(k))
        return k


def rc_kmer(seq):
    """ Return reverse complement of a string (base) sequence. """
    return reduce(lambda x,y: x+y, map(com, seq[::-1]))


def reverse_complement(seq):
    """ Return reverse complement of a string (base) sequence. """
    return rc_kmer(seq)


def shotgun_library(seq, mu, sigma, direction=(1,-1)):
    """Generate random fragment sequences of a given input sequence

    :param seq: input sequence.
    :param mu: mean fragment length.
    :param sigma: stdv of fragment length.
    :param direction: tuple represention direction of output sequences with
        respect to the input sequence.

    .. note:: Could be made more efficient using buffers for random samples
        and handling cases separately.
    """
    seq_len = len(seq)
    while True:
        start = np.random.randint(0, seq_len)
        frag_length = int(np.random.normal(mu, sigma))
        move = np.random.choice(direction)
        end = max(0, start + move*frag_length)
        start, end = sorted([start, end])

        if end - start < 2:
            # Expand a bit to ensure we grab at least one base.
            start = max(0, start - 1)
            end += 1

        frag_seq = seq[start:end]
        if move == -1:
            frag_seq = reverse_complement(frag_seq)
        yield frag_seq


def seq_to_kmers(seq, length):
    """ Turn a string into a list of (overlapping) kmers.

    e.g. perform the transformation:

    'ATATGCG' => ['ATA','TAT', 'ATG', 'TGC', 'GCG']

    :param seq: character string
    :param length: length of kmers in output
    """
    return [seq[x:x+length] for x in range(0, len(seq)-length + 1)]


def kmers_to_annotated_sequence(kmers):
    """ From a sequence of kmers calculate a contiguous symbol string
    and a list indexing the first kmer in which the symbol was observed.

    *Returns* a tuple containing:

    ================  ======================================================
    *sequence*        contiguous symbol string
    *indices*         indices of *kmers* with first occurence of
                      corresponding symbol in *sequence*
    ================  ======================================================
    """
    overlaps = kmer_overlap(kmers)
    sequence = kmers_to_call(kmers, overlaps)
    pos = np.cumsum(overlaps, dtype=int)
    indices = [-1] * len(sequence)
    lastpos = -1
    for i, p in enumerate(pos):
        if p != lastpos:
            indices[p] = i
            lastpos = p
    return sequence, indices


def kmer_overlap(kmers, moves=None, it=False):
    """From a list of kmers return the character shifts between them.
    (Movement from i to i+1 entry, e.g. [AATC,ATCG] returns [0,1]).

    :param kmers: sequence of kmer strings.
    :param moves: allowed movements, if None all movements to length of kmer
        are allowed.
    :param it: yield values instead of returning a list.

    Allowed moves may be specified in moves argument in order of preference.
    """

    if it:
        return kmer_overlap_gen(kmers, moves)
    else:
        return list(kmer_overlap_gen(kmers, moves))


def kmer_overlap_gen(kmers, moves=None):
    """From a list of kmers return the character shifts between them.
    (Movement from i to i+1 entry, e.g. [AATC,ATCG] returns [0,1]).
    Allowed moves may be specified in moves argument in order of preference.

    :param moves: allowed movements, if None all movements to length of kmer
        are allowed.
    """

    first = True
    yield 0
    for last_kmer, this_kmer in window(kmers, 2):
        if first:
            if moves is None:
                l = len(this_kmer)
                moves = range(l + 1)
            first = False

        l = len(this_kmer)
        for j in moves:
            if j < 0:
                if last_kmer[:j] == this_kmer[-j:]:
                    yield j
                    break
            elif j > 0 and j < l:
                if last_kmer[j:l] == this_kmer[0:-j]:
                    yield j
                    break
            elif j == 0:
                if last_kmer == this_kmer:
                    yield 0
                    break
            else:
                yield l
                break


def kmers_to_call(kmers, moves):
    """From a list of kmers and movements, produce a basecall.

    :param kmers: iterable of kmers
    :param moves: iterbale of character overlaps between kmers
    """

    # We use izip longest to check that iterables are same length
    bases = None
    for kmer, move in izip_longest(kmers, moves, fillvalue=None):
        if kmer is None or move is None:
            raise RuntimeError('Lengths of kmers and moves must be equal (kmers={} and moves={}.'.format(len(kmers), len(moves)))
        if move < 0 and not math.isnan(x):
            raise RuntimeError('kmers_to_call() cannot perform call when backward moves are present.')

        if bases  is None:
            bases = kmer
        else:
            if math.isnan(move):
                bases = bases + 'N' + kmer
            else:
                bases = bases + kmer[len(kmer) - int(move):len(kmer)]
    return bases


def kmers_to_sequence(kmers):
    """Convert a sequence of kmers into a contiguous symbol string.

    :param kmers: list of kmers from which to form a sequence

    .. note:
       This is simply a convenient synthesis of :func:`kmer_overlap`
       and :func:`kmers_to_call`
    """
    return kmers_to_call(kmers, kmer_overlap(kmers))


def random_string(length=6):
    """Return a random upper-case string of given length.

    :param length: length of string to return.
    """

    return ''.join(random.choice(string.ascii_uppercase) for _ in range(length))


def conf_line(option, value, pad=30):
    return '{} = {}\n'.format(option.ljust(pad), value)


def window(iterable, size):
    """Create an iterator returning a sliding window from another iterator.

    :param iterable: iterable object.
    :param size: size of window.
    """

    iters = tee(iterable, size)
    for i in xrange(1, size):
        for each in iters[i:]:
            next(each, None)
    return izip(*iters)


def group_by_list(iterable, group_sizes):
    """Yield successive varying size lists from iterator"""
    sizes = cycle(group_sizes)
    it = iter(iterable)
    while True:
        chunk_it = islice(it, sizes.next())
        try:
            first_el = next(chunk_it)
        except StopIteration:
            break
        yield list(chain((first_el,), chunk_it))


def ncycles(iterable, n):
    "Returns the sequence elements n times"
    return chain.from_iterable(repeat(tuple(iterable), n))


class AddFields(object):
    """Helper to add numerous fields to a numpy array. (Syntactic
    sugar around numpy.lib.recfunctions.append_fields)."""
    def __init__(self, array):
        self.array = array
        self.data = []
        self.fields = []
        self.dtypes = []

    def add(self, field, data, dtype=None):
        """Add a field.

        :param field: field name.
        :param data: column of data.
        :param dtype: dtype of data column.
        """
        if len(data) != len(self.array):
            raise TypeError('Length of additional field must be equal to base array.')

        if dtype is None:
            dtype = data.dtype
        self.fields.append(field)
        self.data.append(data)
        self.dtypes.append(dtype)

    def finalize(self):
        return nprf.append_fields(self.array, self.fields, self.data, self.dtypes, usemask=False)


def docstring_parameter(*sub):
    """Allow docstrings to contain parameters."""
    def dec(obj):
        obj.__doc__ = obj.__doc__.format(*sub)
        return obj
    return dec


class FastaWrite(object):
    def __init__(self, filename=None, fastq=False):
        """Simple Fasta writer to file or stdout. The only task this
        class achieves is formatting sequences into fixed line lengths.

        :param filename: if `None` or '-' output is written to stdout
            else it is written to a file opened with name `filename`.
        :param mode: mode for opening file.
        """
        self.filename = filename
        self.fastq = fastq

    def __enter__(self):
        if self.filename is not None and self.filename != '-':
            self.fh = open(self.filename, 'w', 0)
        else:
            self.fh = os.fdopen(sys.stdout.fileno(), 'w', 0)
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        if self.fh is not sys.stdout:
            self.fh.close()

    def write(self, name, seq, qual=None, meta=None, line_length=80):
        if self.fastq:
            self._write_fastq(name, seq, qual, meta)
            return

        #TODO: handle meta
        self.fh.write(">{}\n".format(name))
        for chunk in (seq[i:i+line_length] for i in xrange(0, len(seq), line_length)):
            self.fh.write('{}\n'.format(chunk))
        self.fh.flush()

    def _write_fastq(self, name, seq, qual=None, meta=None):
        if qual is None:
            qual = '!'*len(seq)
        #TODO: handle meta
        self.fh.write("@{}\n{}\n+\n{}\n".format(name, seq, qual))
        self.fh.flush()


def _try_except_pass(func, *args, **kwargs):
    """Implementation of try_except_pass below. When wrapping a function we
    would ordinarily form a closure over a (sub)set of the inputs. Such
    closures cannot be pickled however since the wrapper name is not
    importable. We get around this by using functools.partial (which is
    pickleable). The result is that we can decorate a function to mask
    exceptions thrown by it.
    """

    # Strip out "our" arguments, this slightly perverse business allows
    #    us to call the target function with multiple arguments.
    recover = kwargs.pop('recover', None)
    recover_fail = kwargs.pop('recover_fail', False)
    try:
        return func(*args, **kwargs)
    except:
        exc_info = sys.exc_info()
        try:
            if recover is not None:
                recover(*args, **kwargs)
        except Exception as e:
            sys.stderr.write("Unrecoverable error.")
            if recover_fail:
                raise e
            else:
                traceback.print_exc(sys.exc_info()[2])
        # print the original traceback
        traceback.print_tb(exc_info[2])
        return None


def try_except_pass(func, recover=None, recover_fail=False):
    """Wrap a function to mask exceptions that it may raise. This is
    equivalent to::

        def try_except_pass(func):
            def wrapped()
                try:
                    func()
                except Exception as e:
                    print str(e)
            return wrapped

    in the simplest sense, but the resulting function can be pickled.

    :param func: function to call
    :param recover: function to call immediately after exception thrown in
        calling `func`. Will be passed same args and kwargs as `func`.
    :param recover_fail: raise exception if recover function raises?

    ..note::
        See `_try_except_pass` for implementation, which is not locally
        scoped here because we wish for it to be pickleable.

    ..warning::
        Best practice would suggest this to be a dangerous function. Consider
        rewriting the target function to better handle its errors. The use
        case here is intended to be ignoring exceptions raised by functions
        when mapped over arguments, if failures for some arguments can be
        tolerated.

    """
    return partial(_try_except_pass, func, recover=recover, recover_fail=recover_fail)


class __NotGiven(object):
    def __init__(self):
        """Some horrible voodoo"""
        pass


def tang_imap(
    function, args, fix_args=__NotGiven(), fix_kwargs=__NotGiven(),
    threads=1, unordered=False, chunksize=1,
    pass_exception=False, recover=None, recover_fail=False,
):
    """Wrapper around various map functions

    :param function: the function to apply, must be pickalable for multiprocess
        mapping (problems will results if the function is not at the top level
        of scope).
    :param args: iterable of argument values of function to map over
    :param fix_args: arguments to hold fixed
    :param fix_kwargs: keyword arguments to hold fixed
    :param threads: number of subprocesses
    :param unordered: use unordered multiprocessing map
    :param chunksize: multiprocessing job chunksize
    :param pass_exception: ignore exceptions thrown by function?
    :param recover: callback for recovering from exceptions in function
    :param recover_fail: reraise exceptions when recovery fails?

    .. note::
        This function is a generator, the caller will need to consume this.

    If fix_args or fix_kwargs are given, these are first used to create a
    partially evaluated version of function.

    The special :class:`__NotGiven` is used here to flag when optional arguments
    are to be used.
    """

    my_function = function
    if not isinstance(fix_args, __NotGiven):
        my_function = partial(my_function, *fix_args)
    if not isinstance(fix_kwargs, __NotGiven):
        my_function = partial(my_function, **fix_kwargs)

    if pass_exception:
        my_function = try_except_pass(my_function, recover=recover, recover_fail=recover_fail)

    if threads == 1:
        for r in imap(my_function, args):
            yield r
    else:
        pool = Pool(threads)
        if unordered:
            mapper = pool.imap_unordered
        else:
            mapper = pool.imap
        for r in mapper(my_function, args, chunksize=chunksize):
            yield r
        pool.close()
        pool.join()


def fileno(file_or_fd):
    """Return a file descriptor from a file descriptor or file object."""
    fd = getattr(file_or_fd, 'fileno', lambda: file_or_fd)()
    if not isinstance(fd, int):
        raise ValueError("Expected a file (`.fileno()`) or a file descriptor")
    return fd


@contextmanager
def stderr_redirected(to=os.devnull, stderr=sys.stderr):
    """Redirect stderr (optionally something else) at the file
    descriptor level. Defaults allow ignoring of stderr.

    :param to: redirection target
    :param stderr: stream to redirect
    """
    stderr_fd = fileno(stderr)
    # copy stderr_fd before it is overwritten
    #NOTE: `copied` is inheritable on Windows when duplicating a standard stream
    with os.fdopen(os.dup(stderr_fd), 'wb') as copied:
        stderr.flush()  # flush library buffers that dup2 knows nothing about
        try:
            os.dup2(fileno(to), stderr_fd)  # $ exec 2>&to
        except ValueError:  # filename
            with open(to, 'wb') as to_file:
                os.dup2(to_file.fileno(), stderr_fd)  # $ exec 2> to
        try:
            yield stderr # allow code to be run with the redirected stderr
        finally:
            # restore stderr to its previous value
            #NOTE: dup2 makes stderr_fd inheritable unconditionally
            stderr.flush()
            os.dup2(copied.fileno(), stderr_fd)  # $ exec 2>&copied
