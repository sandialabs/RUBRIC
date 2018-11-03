import itertools
import numpy as np

from numpy.ctypeslib import ndpointer
from ctypes import c_size_t

from RUBRIC.nanonet import cl
from RUBRIC.nanonet import dtype, tiny, build_program
from RUBRIC.nanonet import get_shared_lib

nanonetdecode = get_shared_lib('nanonetdecode')

_ETA = tiny
_BASES = ['A', 'C', 'G', 'T']
_DIBASES = [b1 + b2 for b1 in _BASES for b2 in _BASES]
_NSTEP = len(_BASES)
_NSKIP = _NSTEP ** 2
_STEP_FACTOR = np.log(_NSTEP)
_SKIP_FACTOR = np.log(_NSKIP)


def decode_profile(post, trans=None, log=False, slip=0.0):
    """  Viterbi-style decoding with per-event transition weights
    (profile)
    :param post: posterior probabilities of kmers by event.
    :param trans: A generator (e.g. a :class:`ndarray`) to produce
    per-transition log-scaled weights. None == no transition weights.
    :param log: Posterior probabilities are in log-space.
    """
    nstate = post.shape[1]
    lpost = post.copy()
    if not log:
        np.add(_ETA, lpost, lpost)
        np.log(lpost, lpost)
    
    if trans is None:
        trans = itertools.repeat(np.zeros(3))
    else:
        trans = np.copy(trans)
        trans[:,1] -= _STEP_FACTOR
        trans[:,2] -= _SKIP_FACTOR

    log_slip = np.log(_ETA + slip)

    pscore = lpost[0]
    trans_iter = trans.__iter__()
    for ev in range(1, len(post)):
        # Forward Viterbi iteration
        ev_trans = trans_iter.next()
        # Stay
        score = pscore + ev_trans[0]
        iscore = range(nstate)
        # Slip
        scoreNew = np.amax(pscore) + log_slip
        iscoreNew = np.argmax(pscore)
        iscore = np.where(score > scoreNew, iscore, iscoreNew)
        score = np.fmax(score, scoreNew)
        # Step
        pscore = pscore.reshape((_NSTEP, -1))
        nrem = pscore.shape[1]
        scoreNew = np.repeat(np.amax(pscore, axis=0), _NSTEP) + ev_trans[1]
        iscoreNew = np.repeat(nrem * np.argmax(pscore, axis=0) + range(nrem), _NSTEP)
        iscore = np.where(score > scoreNew, iscore, iscoreNew)
        score = np.fmax(score, scoreNew)
        # Skip
        pscore = pscore.reshape((_NSKIP, -1))
        nrem = pscore.shape[1]
        scoreNew = np.repeat(np.amax(pscore, axis=0), _NSKIP) + ev_trans[2]
        iscoreNew = np.repeat(nrem * np.argmax(pscore, axis=0) + range(nrem), _NSKIP)
        iscore = np.where(score > scoreNew, iscore, iscoreNew)
        score = np.fmax(score, scoreNew)
        # Store
        lpost[ev-1] = iscore
        pscore = score + lpost[ev]

    state_seq = np.zeros(len(post), dtype=int)
    state_seq[-1] = np.argmax(pscore)
    for ev in range(len(post), 1, -1):
        # Viterbi backtrace
        state_seq[ev-2] = int(lpost[ev-2][state_seq[ev-1]])

    return np.amax(pscore), state_seq
    
def decode_profile_opencl(ctx, queue_list, post_list, trans_list=None, log=False, slip=0.0, max_workgroup_size=256):
    """  Viterbi-style decoding with per-event transition weights
    (profile)
    :param post: posterior probabilities of kmers by event.
    :param trans: A generator (e.g. a :class:`ndarray`) to produce
    per-transition log-scaled weights. None == no transition weights.
    :param log: Posterior probabilities are in log-space.
    """

    fp_type = np.float32 # uses this floating point type in the kernel
    iter = len(queue_list)
    
    if trans_list is None:
        for x in xrange(iter):
            trans_list.append(itertools.repeat(np.zeros(3)))
    else:
        for x in xrange(iter):
            trans = trans_list[x]
            trans = np.copy(trans)
            trans[:,1] -= _STEP_FACTOR
            trans[:,2] -= _SKIP_FACTOR
            trans_list[x] = trans
    
    lpostList = []
    if fp_type == np.float32:
        slip = np.float32(slip)
        for x in xrange(iter):
            lpostList.append(post_list[x].copy().astype(np.float32))
            trans_list[x] = np.float32(trans_list[x])
    else:
        slip = np.float64(slip)
        for x in xrange(iter):
            lpostList.append(post_list[x].copy())
    
    cl_postList = []
    cl_transList = []
    cl_state_seqList = []
    cl_pscore_maxList = []
    state_seqList = []
    pscore_max = np.zeros(1, dtype=fp_type)
    for x in xrange(iter):
        cl_postList.append(cl.Buffer(ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.USE_HOST_PTR, hostbuf=np.ravel(lpostList[x])))
        cl_transList.append(cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.USE_HOST_PTR, hostbuf=np.ravel(trans_list[x])))
        state_seqList.append(np.zeros(len(lpostList[x]), dtype=np.int32))
        cl_state_seqList.append(cl.Buffer(ctx, cl.mem_flags.READ_WRITE, len(lpostList[x])*state_seqList[x].itemsize))
        cl_pscore_maxList.append(cl.Buffer(ctx, cl.mem_flags.READ_WRITE, 1*pscore_max.itemsize))

    local_x = global_x = max_workgroup_size
    local_y = global_y = 1
    
    prg = build_program(ctx, kernel_code, extra=
    " -DWORK_ITEMS={} -DNUM_STATES={}".format(
        local_x, lpostList[0].shape[1]
    ))

    for x in xrange(iter):
        prg.decode(queue_list[x], (global_x, global_y), (local_x, local_y), np.int32(lpostList[x].shape[0]), slip, cl_postList[x], cl_transList[x], cl_state_seqList[x], cl_pscore_maxList[x])
        queue_list[x].flush()
    
    pscore_maxList = []
    for x in xrange(iter):
        cl.enqueue_copy(queue_list[x], pscore_max, cl_pscore_maxList[x])
        pscore_maxList.append(pscore_max[0])
        cl.enqueue_copy(queue_list[x], state_seqList[x], cl_state_seqList[x])

    return pscore_maxList, state_seqList

def decode_transition(post, trans, log=False, slip=0.0):
    """  Viterbi-style decoding with weighted transitions
    :param post: posterior probabilities of kmers by event.
    :param trans: (log) penalty for [stay, step, skip]
    :param log: Posterior probabilities are in log-space.
    """
    return decode_profile(post, trans=itertools.repeat(trans), log=log, slip=slip)


def decode_simple(post, log=False, slip=0.0):
    """  Viterbi-style decoding with uniform transitions
    :param post: posterior probabilities of kmers by event.
    :param log: Posterior probabilities are in log-space.
    """
    return decode_profile(post, log=log, slip=slip)


def decode_homogenous(post, log=False, n_bases=4):
    if not log:
        post = np.log(_ETA + post)
    post = post.astype(np.float32)

    func = nanonetdecode.decode_path
    func.restype = np.float32
    func.argtypes = [ndpointer(dtype=np.float32, flags='CONTIGUOUS'),
                     c_size_t, c_size_t, c_size_t]
    score = func(post, post.shape[0], n_bases, post.shape[1])
    states = post[:,0].astype(int)
    return score, states


def estimate_transitions(post, trans=None):
    """  Naive estimate of transition behaviour from posteriors
    :param post: posterior probabilities of kmers by event.
    :param trans: prior belief of transition behaviour (None = use global estimate)
    """
    assert trans is None or len(trans) == 3, 'Incorrect number of transitions'
    res = np.zeros((len(post), 3))
    res[:] = _ETA

    for ev in range(1, len(post)):
        stay = np.sum(post[ev-1] * post[ev])
        p = post[ev].reshape((-1, _NSTEP))
        step = np.sum(post[ev-1] * np.tile(np.sum(p, axis=1), _NSTEP)) / _NSTEP
        p = post[ev].reshape((-1, _NSKIP))
        skip = np.sum(post[ev-1] * np.tile(np.sum(p, axis=1), _NSKIP)) / _NSKIP
        res[ev-1] = [stay, step, skip]

    if trans is None:
        trans = np.sum(res, axis=0)
        trans /= np.sum(trans)

    res *= trans
    res /= np.sum(res, axis=1).reshape((-1,1))

    return res

def fast_estimate_transitions(post, trans=None, n_bases=4):
    """  Naive estimate of transition behaviour from posteriors
    :param post: posterior probabilities of kmers by event.
    :param trans: prior belief of transition behaviour (None = use global estimate)
    """

    assert trans is None or len(trans) == 3, 'Incorrect number of transitions'
    res = np.empty((len(post), 3), dtype=dtype)
    res.fill(_ETA)

    func = nanonetdecode.estimate_transitions
    func.restype = None
    func.argtypes = [ndpointer(dtype=np.float32, flags='CONTIGUOUS'), 
                     ndpointer(dtype=np.float32, flags='CONTIGUOUS'),
                     c_size_t, c_size_t, c_size_t]
    func(post, res, post.shape[0], n_bases, post.shape[1])

    if trans is None:
        trans = np.sum(res, axis=0)
        trans /= np.sum(trans)

    res *= trans
    res /= np.sum(res, axis=1).reshape((-1,1))
    return res


kernel_code = """
#if __OPENCL_VERSION__ <= CL_VERSION_1_1
#pragma OPENCL EXTENSION cl_khr_fp64: enable
#endif

#if FPTYPE == float
#define _ETA 1e-30F
#define FPTYPE_MAX FLT_MAX
#else
#define _ETA 1e-300
#define FPTYPE_MAX DBL_MAX
#endif

#define _NSTEP 4
#define _NSKIP (_NSTEP*_NSTEP)

inline void max_element(
    __local FPTYPE* buffer,
    __local int* ibuffer,
    int stop
    ) 
{
    // Perform parallel reduction
    int local_index = get_local_id(0);
    for(int offset = get_local_size(0) / 2; offset > stop; offset = offset / 2) 
    {
        if (local_index < offset) 
        {
            if(buffer[local_index] < buffer[local_index + offset])
            {
                buffer[local_index] = buffer[local_index + offset];
                ibuffer[local_index] = ibuffer[local_index + offset];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
} 

__kernel __attribute__((reqd_work_group_size(WORK_ITEMS, 1, 1))) 
void decode(
    int size,
    FPTYPE slip,
    __global FPTYPE* restrict post, 
    __global const FPTYPE* restrict trans,
    __global int* restrict state_seq,
    __global FPTYPE* restrict pscore_max
) {
    int local_id = get_local_id(0);
    int local_size = get_local_size(0);
    
    __local FPTYPE buffer[WORK_ITEMS];
    __local int ibuffer[WORK_ITEMS];
    
    FPTYPE score[NUM_STATES/WORK_ITEMS];
    int iscore[NUM_STATES/WORK_ITEMS];
    
    FPTYPE score_new = -FPTYPE_MAX;
    int iscore_new = 0;
    const FPTYPE log_slip = log(slip + _ETA);
    
    for(int x = 0; x < size; ++x)
        for(int y = 0; y < NUM_STATES; y += local_size)
            post[x*NUM_STATES + y + local_id] = log(post[x*NUM_STATES + y + local_id] + _ETA);
    
    for(int x = 1; x < size; ++x)
    {
        FPTYPE trans0 = trans[(x-1)*3];
        FPTYPE trans1 = trans[(x-1)*3+1];
        FPTYPE trans2 = trans[(x-1)*3+2];
        
        // slip
        score_new = -FPTYPE_MAX;
        iscore_new = 0;
        for(int y = 0; y < NUM_STATES; y += local_size)
        {
            score[y/WORK_ITEMS] = buffer[local_id] = post[(x-1)*NUM_STATES + y + local_id];
            iscore[y/WORK_ITEMS] = ibuffer[local_id] = y + local_id;
            barrier(CLK_LOCAL_MEM_FENCE); 
            
            max_element(buffer, ibuffer, 0);
            if(buffer[0] > score_new)
            {
                score_new = buffer[0];
                iscore_new = ibuffer[0];
            }
            barrier(CLK_LOCAL_MEM_FENCE);            
        }
        score_new += log_slip;
        
        // stay
        for(int y = 0; y < NUM_STATES; y += local_size)
        {
            score[y/WORK_ITEMS] += trans0;
            //iscore[y/WORK_ITEMS] = y + local_id;
            if(score_new >= score[y/WORK_ITEMS])
            {
                score[y/WORK_ITEMS] = score_new;
                iscore[y/WORK_ITEMS] = iscore_new;
            }  
        }
            
        // step
        for(int y = 0; y < NUM_STATES; y += local_size)
        {
            buffer[local_id] = post[(x-1)*NUM_STATES + ((y/local_size)*(WORK_ITEMS/4)) + ((local_id/(WORK_ITEMS/4))*(NUM_STATES/4)) + (local_id%(WORK_ITEMS/4))];
            ibuffer[local_id] = (y/local_size)*(WORK_ITEMS/4) + ((local_id/(WORK_ITEMS/4))*(NUM_STATES/4)) + (local_id%(WORK_ITEMS/4));
            barrier(CLK_LOCAL_MEM_FENCE);
            
            max_element(buffer, ibuffer, WORK_ITEMS/8);
            buffer[local_id] += trans1;
            barrier(CLK_LOCAL_MEM_FENCE);
            if(buffer[local_id/4] > score[y/WORK_ITEMS])
            {
                score[y/WORK_ITEMS] = buffer[local_id/4];
                iscore[y/WORK_ITEMS] = ibuffer[local_id/4];
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        
        // skip
        for(int y = 0; y < NUM_STATES; y += local_size)
        {
            buffer[local_id] = post[(x-1)*NUM_STATES + ((y/local_size)*(WORK_ITEMS/16)) + ((local_id/(WORK_ITEMS/16))*(NUM_STATES/16)) + (local_id%(WORK_ITEMS/16))];
            ibuffer[local_id] = (y/local_size)*(WORK_ITEMS/16) + ((local_id/(WORK_ITEMS/16))*(NUM_STATES/16)) + (local_id%(WORK_ITEMS/16));
            barrier(CLK_LOCAL_MEM_FENCE);
            
            max_element(buffer, ibuffer, WORK_ITEMS/32);
            buffer[local_id] += trans2;
            barrier(CLK_LOCAL_MEM_FENCE);
            if(buffer[local_id/16] > score[y/WORK_ITEMS])
            {
                score[y/WORK_ITEMS] = buffer[local_id/16];
                iscore[y/WORK_ITEMS] = ibuffer[local_id/16];
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        
        for(int y = 0; y < NUM_STATES; y += local_size)
        {
            post[(x-1)*NUM_STATES + y + local_id] = iscore[y/WORK_ITEMS];
            post[x*NUM_STATES + y + local_id] += score[y/WORK_ITEMS];
        }
    }
    
    score_new = -FPTYPE_MAX;
    iscore_new = 0;
    for(int y = 0; y < NUM_STATES; y += local_size)
    {
        buffer[local_id] = post[(size-1)*NUM_STATES + y + local_id];
        ibuffer[local_id] = y + local_id;
        barrier(CLK_LOCAL_MEM_FENCE); 
        
        max_element(buffer, ibuffer, 0);
        if(buffer[0] > score_new)
        {
            score_new = buffer[0];
            iscore_new = ibuffer[0];
        }
        barrier(CLK_LOCAL_MEM_FENCE);            
    }

    if(local_id == 0)
    {
        state_seq[size-1] = iscore_new;
        for(int x = size; x > 1; --x)
            state_seq[x-2] = post[(x-2)*NUM_STATES + state_seq[x-1]];
        *pscore_max = score_new; 
    }
    
}
"""
