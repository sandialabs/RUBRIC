import abc
import numpy as np

from RUBRIC.nanonet import cl

dtype = np.float32
tiny = np.finfo(dtype).tiny

def tanh(x):
    return np.tanh(x)

def tanh_approx(x):
    """Pade approximation of tanh function
    http://musicdsp.org/archive.php?classid=5#238
    """
    xsqr = np.square(x)
    tanh_p = x * (27.0 + xsqr) / (27.0 + 9.0 * xsqr)
    return np.clip(tanh_p, -1.0, 1.0)

def sigmoid(x):
    return np.reciprocal(1.0 + np.exp(-x))

def sigmoid_approx(x):
   """Approximation of sigmoid function
   https://github.com/Theano/Theano/blob/master/theano/tensor/nnet/sigm.py#L217
   """
   xabs = np.fabs(x)
   tmp = np.where(xabs < 3.0, 0.4677045353015495 + 0.02294064733985825 * (xabs - 1.7), 0.497527376843365)
   tmp = np.where(xabs < 1.7, 0.75 * xabs / (1.0 + xabs), tmp)
   return np.sign(x) * tmp + 0.5

def linear(x):
    return x

def softplus(x):
    return np.log1p(np.exp(x))

def relu(x):
    return np.where(x > 0.0, x, 0.0)


class Layer(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def run(self, inMat):
        """Run network layer"""
        pass

    @abc.abstractproperty
    def in_size(self):
        """Input size"""
        pass

    @abc.abstractproperty
    def out_size(self):
        """Output size"""
        pass


class RNN(Layer):
    @abc.abstractmethod
    def step(self, in_vec, state):
        """A single step along the RNN.

        :param in_vec: Input to node
        :param state: Hidden state from previous node
        """
        pass


class FeedForward(Layer):
    """Basic feedforward layer

        out = f(inMat W + b)

    :param W: Weight matrix of dimension (|input|, size)
    :param b: Bias vector of length size. Optional with default of no bias.
    :param fun: The activation function.  Must accept a numpy array as input.
    """
    def __init__(self, W, b=None, fun=tanh):
        assert b is None or len(b) == W.shape[1]
        self.b = np.zeros(W.shape[1], dtype=dtype) if b is None else b
        self.W = W.astype(dtype)
        self.f = fun

    def __setstate__(self, d):
        self.__dict__ = d
        for attr in ('W', 'b'):
            setattr(self, attr, getattr(self, attr).astype(dtype))

    @property
    def in_size(self):
        return self.W.shape[0]

    @property
    def out_size(self):
        return self.W.shape[1]

    def run(self, inMat, ctx=None, queueList=None):
        if queueList is not None:
            return self._run_opencl(inMat, ctx, queueList)
        else:
            assert self.in_size == inMat.shape[1]
            return self.f(inMat.dot(self.W) + self.b)

    def _run_opencl(self, inMat, ctx, queueList):
        for mat in inMat:
            assert self.in_size == mat.shape[1]
        iter = len(inMat)
        
        kernel_src = kernel_code_feedforward
        
        # Calculate work items 
        local_x = 256
        local_y = 1
        global_x_list = []
        for mat in inMat:
            global_x = mat.shape[0]
            if global_x % local_x:
                global_x = (global_x / local_x + 1) * local_x 
            global_x_list.append(global_x)
        global_y = 1
        
        # Build the kernel (builds for the first time, then uses cached version)
        prg = build_program(ctx, kernel_src, extra=
            "-DWORK_ITEMS={} -DIN_MAT_Y={}".format(
            local_x, inMat[0].shape[1]
        ))

        Wtr = np.transpose(self.W)
        
        # Allocate OpenCL buffers    
        cl_inMatList = []
        buffer_flags = cl.mem_flags.READ_ONLY | cl.mem_flags.USE_HOST_PTR
        for x in xrange(iter):
            cl_inMatList.append(cl.Buffer(ctx, buffer_flags, hostbuf=np.ravel(inMat[x])))
        cl_W = cl.Buffer(ctx, buffer_flags, hostbuf=np.ravel(Wtr))
        cl_b = cl.Buffer(ctx, buffer_flags, hostbuf=np.ravel(self.b))
        cl_outList = []
        for x in xrange(iter):
            cl_outList.append(cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, inMat[x].shape[0]*self.W.shape[1]*inMat[x].itemsize))

        # Run the kernel
        for x in xrange(iter):
            prg.run_layer(queueList[x], (global_x_list[x], global_y), (local_x, local_y), np.int32(inMat[x].shape[0]), np.int32(Wtr.shape[0]), cl_inMatList[x], cl_W, cl_b, cl_outList[x])
            queueList[x].flush()
        
        # Copy results back to host (blocking call)
        outList = []
        for x in xrange(iter):
            outList.append(np.zeros((inMat[x].shape[0],self.W.shape[1]), dtype=dtype))
            cl.enqueue_copy(queueList[x], outList[x], cl_outList[x])
        return outList
            

class SoftMax(Layer):
    """Softmax layer

        tmp = exp(inmat W + b)
        out = row_normalise(tmp)

    :param W: Weight matrix of dimension (|input|, size)
    :param b: Bias vector of length size.  Optional with default of no bias.
    """
    def __init__(self, W, b=None):
        assert b is None or len(b) == W.shape[1]
        self.b = np.zeros(W.shape[1], dtype=dtype) if b is None else b
        self.W = W.astype(dtype)

    def __setstate__(self, d):
        self.__dict__ = d
        for attr in ('W', 'b'):
            setattr(self, attr, getattr(self, attr).astype(dtype))

    @property
    def in_size(self):
        return self.W.shape[0]

    @property
    def out_size(self):
        return self.W.shape[1]

    def run(self, inMat, ctx=None, queueList=None):
        if queueList is not None:
            return self._run_opencl(inMat, ctx, queueList)
        else:
            assert self.in_size == inMat.shape[1]
            tmp =  inMat.dot(self.W) + self.b
            m = np.amax(tmp, axis=1).reshape((-1,1))
            tmp = np.exp(tmp - m)
            x = np.sum(tmp, axis=1)
            tmp /= x.reshape((-1,1))
            return tmp

    def _run_opencl(self, inMat, ctx, queueList):
        for mat in inMat:
            assert self.in_size == mat.shape[1]
        iter = len(inMat)
        
        fp_type = dtype
        kernel_src = kernel_code_softmax
        
        # Calculate work items 
        local_x = 256
        local_y = 1
        global_x_list = []
        for mat in inMat:
            global_x = mat.shape[0]
            if global_x % local_x:
                global_x = (global_x / local_x + 1) * local_x 
            global_x_list.append(global_x) 
        global_y = 1
        local_x_softmax = 256
        
        # Build the kernel (builds for the first time, then uses cached version)
        prg = build_program(ctx, kernel_src, extra=
            '-DWORK_ITEMS={} -DIN_MAT_Y={} -DWORK_ITEMS_PAR={} -DITER={}'.format(
            local_x, inMat[0].shape[1], local_x_softmax, (self.W.shape[1]-1) / local_x_softmax
        ))

        Wtr = np.transpose(self.W)
        
        # Allocate OpenCL buffers    
        cl_inMatList = []
        buffer_flags = cl.mem_flags.READ_ONLY | cl.mem_flags.USE_HOST_PTR
        for x in xrange(iter):
            cl_inMatList.append(cl.Buffer(ctx, buffer_flags, hostbuf=np.ravel(inMat[x])))
        cl_W = cl.Buffer(ctx, buffer_flags, hostbuf=np.ravel(Wtr))
        cl_b = cl.Buffer(ctx, buffer_flags, hostbuf=np.ravel(self.b))
        cl_outList = []
        for x in xrange(iter):
            cl_outList.append(cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, inMat[x].shape[0]*self.W.shape[1]*inMat[x].itemsize))

        # Run the kernel
        for x in xrange(iter):
            prg.run_layer(queueList[x], (global_x_list[x], global_y), (local_x, local_y), np.int32(inMat[x].shape[0]), np.int32(Wtr.shape[0]), cl_inMatList[x], cl_W, cl_b, cl_outList[x])
            queueList[x].flush()
        for x in xrange(iter):
            prg.run_softmax(queueList[x], (inMat[x].shape[0]*local_x_softmax, 1), (local_x_softmax, 1), np.int32(Wtr.shape[0]), cl_outList[x])
            queueList[x].flush()
        
        # Copy results back to host (blocking call)
        outList = []
        for x in xrange(iter):
            outList.append(np.zeros((inMat[x].shape[0],self.W.shape[1]), dtype=dtype))
            cl.enqueue_copy(queueList[x], outList[x], cl_outList[x])
        return outList


class SimpleRNN(RNN):
    """A simple recurrent layer

        Step: state_new = fun([state_old, input_new] W + b)
              output_new = state_new

    :param W: Weight matrix of dimension (|input| + size, size)
    :param b: Bias vector of length  size.  Optional with default of no bias.
    :param fun: The activation function.  Must accept a numpy array as input.
    """
    def __init__(self, W, b=None, fun=tanh):
        assert W.shape[0] > W.shape[1]
        assert b is None or len(b) == W.shape[1]
        self.b = np.zeros(W.shape[1], dtype=dtype) if b is None else b
        self.W = W.astype(dtype)

        self.fun = fun
        self.size = W.shape[0] - W.shape[1]

    def __setstate__(self, d):
        self.__dict__ = d
        for attr in ('W', 'b'):
            setattr(self, attr, getattr(self, attr).astype(dtype))

    @property
    def in_size(self):
        return self.size

    @property
    def out_size(self):
        return self.W.shape[1]

    def step(self, in_vec, state):
        state_out = self.fun(np.concatenate((state, in_vec)).dot(self.W) + self.b)
        return state_out

    def run(self, inMat, ctx=None, queueList=None):
        assert self.in_size == inMat.shape[1]
        out = np.zeros((inMat.shape[0], self.out_size), dtype=dtype)
        state = np.zeros(self.out_size, dtype=dtype)
        for i, v in enumerate(inMat):
            state = self.step(v, state)
            out[i] = state
        return out


class LSTM(RNN):
    def __init__(self, iW, lW, b=None, p=None):
        """Long short-term memory layer with peepholes. Implementation is to be consistent with
        Currennt and may differ from other descriptions of LSTM networks (e.g.
        http://colah.github.io/posts/2015-08-Understanding-LSTMs/).

            Step:
                v = [ input_new, output_old ]
                Pforget = sigmoid( v W2 + b2 + state * p1)
                Pupdate = sigmoid( v W1 + b1 + state * p0)
                Update  = tanh( v W0 + b0 )
                state_new = state_old * Pforget + Update * Pupdate
                Poutput = sigmoid( v W3 + b3 + state * p2)
                output_new = tanh(state) * Poutput

        :param iW: weights for cells taking input from preceeding layer.
            Size (4, -1, size)
        :param lW: Weights for connections within layer
            Size (4, size, size )
        :param b: Bias weights for cells taking input from preceeding layer.
            Size (4, size)
        :param p: Weights for peep-holes
            Size (3, size)
        """
        assert len(iW.shape) == 3 and iW.shape[0] == 4
        size = self.size = iW.shape[2]
        assert lW.shape == (4, size, size)
        if b is None:
            b = np.zeros((4, size), dtype=dtype)
        assert b.shape == (4, size)
        if p is None:
            p = np.zeros((3, size), dtype=dtype)
        assert p.shape == (3, size)

        self.iW = np.ascontiguousarray(iW.transpose((1,0,2)).reshape((-1, 4 * size)), dtype=dtype)
        self.lW = np.ascontiguousarray(lW.transpose((1,0,2)).reshape((size, 4 * size)), dtype=dtype)
        self.b = np.ascontiguousarray(b, dtype=dtype).reshape(-1)
        self.p = np.ascontiguousarray(p, dtype=dtype)
        self.isize = iW.shape[1]

    def __setstate__(self, d):
        self.__dict__ = d
        for attr in ('iW', 'lW', 'b', 'p'):
            setattr(self, attr, getattr(self, attr).astype(dtype))

    @property
    def in_size(self):
        return self.isize

    @property
    def out_size(self):
        return self.size

    def step(self, in_vec, in_state):
        vW = in_vec.dot(self.iW)
        out_prev, prev_state = in_state
        outW = out_prev.dot(self.lW)
        sumW = vW + outW  + self.b
        sumW = sumW.reshape((4, self.size))

        #  Forget gate activation
        state = prev_state * sigmoid(sumW[2] + prev_state * self.p[1] )
        #  Update state with input
        state += tanh(sumW[0]) * sigmoid(sumW[1] + prev_state * self.p[0])
        #  Output gate activation
        out = tanh(state) * sigmoid(sumW[3]  + state * self.p[2])
        return out, state
        
    def run(self, inMat, ctx=None, queueList=None):
        if queueList is not None:
            return self._run_opencl(inMat, ctx, queueList)
        else:
            assert self.in_size == inMat.shape[1]

            out = np.zeros((inMat.shape[0], self.out_size), dtype=dtype)
            out_prev = np.zeros(self.out_size, dtype=dtype)
            state = np.zeros(self.out_size, dtype=dtype)
    
            for i, v in enumerate(inMat):
                out_prev, state = self.step(v, (out_prev, state))
                out[i] = out_prev
            return out

    def _run_opencl(self, inMat, ctx, queueList):
        for mat in inMat:
            assert self.in_size == mat.shape[1]
        iter = len(inMat)
        
        outList = []
        for x in xrange(iter):
            outList.append(np.zeros((inMat[x].shape[0], self.out_size), dtype=dtype))
            
        kernel_src = kernel_code_lstm
        
        # Build the kernel (builds for the first time, then uses cached version)
        prg = build_program(ctx, kernel_src, extra=
            '-DWORK_ITEMS={} -DL_WX={} -DL_WY={}'.format(
            self.iW.shape[1], self.lW.shape[0], self.lW.shape[1]
        ))
        
        # Allocate OpenCL buffers
        cl_inMatList = []
        buffer_flags = cl.mem_flags.READ_ONLY | cl.mem_flags.USE_HOST_PTR
        for x in xrange(iter):
            cl_inMatList.append(cl.Buffer(ctx, buffer_flags, hostbuf=np.ravel(inMat[x])))
        cl_iW = cl.Buffer(ctx, buffer_flags, hostbuf=np.ravel(self.iW))
        cl_lW = cl.Buffer(ctx, buffer_flags, hostbuf=np.ravel(self.lW))
        cl_b  = cl.Buffer(ctx, buffer_flags, hostbuf=self.b)
        cl_p  = cl.Buffer(ctx, buffer_flags, hostbuf=np.ravel(self.p))
        cl_outList = []
        cl_outvW = []
        for x in xrange(iter):
            cl_outList.append(cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, outList[x].shape[0]*self.out_size*inMat[x].itemsize))
            cl_outvW.append(cl.Buffer(ctx, cl.mem_flags.READ_WRITE, outList[x].shape[0]*self.iW.shape[1]*inMat[x].itemsize))

        # Run the kernel
        for x in xrange(iter):
            prg.run_dot(queueList[x], (inMat[x].shape[0]*self.iW.shape[1], 1), (self.iW.shape[1], 1), np.int32(self.iW.shape[0]), np.int32(self.iW.shape[1]), cl_inMatList[x], cl_iW, cl_outvW[x])
            queueList[x].flush()
        for x in xrange(iter):
            prg.run_lstm_layer(queueList[x], (self.iW.shape[1], 1), (self.iW.shape[1], 1), np.int32(inMat[x].shape[0]), cl_outvW[x], cl_lW, cl_b, cl_p, cl_outList[x])
            queueList[x].flush()
        
        # Copy results back to host (blocking call)
        for x in xrange(iter):
            outRavel = np.ravel(outList[x])
            cl.enqueue_copy(queueList[x], outRavel, cl_outList[x])
            outList[x] = np.copy(np.reshape(outRavel, (outList[x].shape[0], outList[x].shape[1])))
        return outList


class Reverse(Layer):
    """Runs a recurrent layer in reverse time (backwards)."""
    def __init__(self, layer):
       self.layer = layer

    @property
    def in_size(self):
        return self.layer.in_size

    @property
    def out_size(self):
        return self.layer.out_size

    def run(self, inMat, ctx=None, queueList=None):
        if not queueList:
            assert self.in_size == inMat.shape[1]
            return self.layer.run(inMat[::-1])[::-1]
        else:
            inMatList = []
            for mat in inMat:
                assert self.in_size == mat.shape[1]
                inMatList.append(mat[::-1])            
            postList= self.layer.run(inMatList, ctx, queueList)
            postListTmp = []
            for post in postList:
                postListTmp.append(post[::-1])
            return postListTmp 


class Parallel(Layer):
    """Run multiple layers in parallel (all have same input and outputs are
    concatenated).
    """
    def __init__(self, layers):
        in_size = layers[0].in_size
        for i in range(1, len(layers)):
            assert in_size == layers[i].in_size, "Incompatible shapes: {} -> {} in layers {}.\n".format(in_size, layers[i].in_size, i)
        self.layers = layers

    @property
    def in_size(self):
        return self.layers[0].in_size

    @property
    def out_size(self):
        return sum(x.out_size for x in self.layers)

    def run(self, inMat, ctx=None, queueList=None):
        if not queueList:
            assert self.in_size == inMat.shape[1]
            return np.hstack(map(lambda x: x.run(inMat), self.layers))
        else:
            for mat in inMat:
                assert self.in_size == mat.shape[1]
            tmp = map(lambda x: x.run(inMat, ctx, queueList), self.layers)
            tmp2 = map(list, zip(*tmp))
            tmp3 = []
            for t in tmp2:
                tmp3.append(np.hstack(t))
            return tmp3 


class Serial(Layer):
    """Run multiple layers serially: output of a layer is the input for the
    next layer.
    """
    def __init__(self, layers):
        prev_out_size = layers[0].out_size
        for i in range(1, len(layers)):
            assert prev_out_size == layers[i].in_size, "Incompatible shapes: {} -> {} in layers {}.\n".format(prev_out_size, layers[i].in_size, i)
            prev_out_size = layers[i].out_size
        self.layers = layers

    @property
    def in_size(self):
        return self.layers[0].in_size

    @property
    def out_size(self):
        return self.layers[-1].out_size

    def run(self, inMat, ctx=None, queueList=None):
        if not queueList:
            assert self.in_size == inMat.shape[1]
            tmp = inMat
            for layer in self.layers:
                tmp = layer.run(tmp)
            return tmp
        else:
            for mat in inMat:
                assert self.in_size == mat.shape[1]
            tmp = inMat
            for layer in self.layers:
                tmp = layer.run(tmp, ctx, queueList)
            return tmp


class BiRNN(Parallel):
    """A bidirectional RNN from two RNNs."""
    def __init__(self, layer1, layer2):
        super(BiRNN, self).__init__((layer1, Reverse(layer2)))


def build_program(ctx, src, extra=None, debug=False):
    fptype = 'double'
    fptype_suffix = ''
    if dtype == np.float32:
        fptype = 'float'
        fptype_suffix = 'f'

    # clang doesn't like macros being used as float suffix
    src = src.replace('FSUF', fptype_suffix)

    if debug:
        for i, line in enumerate(src.splitlines()):
            print '{:03d}{}'.format(i, line)
 
    build_line = '-I. -DFPTYPE={} -DFSUF={}'.format(fptype, fptype_suffix)
    if extra is not None:
        build_line = '{} {}'.format(build_line, extra)
    if debug:
        build_line += ' -Werror'

    return cl.Program(ctx, src).build(build_line)



kernel_code_feedforward = """
#if __OPENCL_VERSION__ <= CL_VERSION_1_1
#pragma OPENCL EXTENSION cl_khr_fp64: enable
#endif

__kernel __attribute__((reqd_work_group_size(WORK_ITEMS, 1, 1)))
__kernel void run_layer(
    int inMatx,
    int Wx, 
    __global const FPTYPE* restrict inMat, 
    __global const FPTYPE* restrict W, 
    __global const FPTYPE* restrict b, 
    __global FPTYPE* restrict ret
){
    int id = get_global_id(0);
    if(id < inMatx)
    {
        FPTYPE inMatBuffer[IN_MAT_Y];
        for(int z = 0; z < IN_MAT_Y; ++z)
            inMatBuffer[z] = inMat[id*IN_MAT_Y+z];
        
        for(int y = 0; y < Wx; ++y)
        {
            FPTYPE r = 0.0FSUF;
            for(int z = 0; z < IN_MAT_Y; ++z)
                r += inMatBuffer[z] * W[y*IN_MAT_Y+z];
            ret[id*Wx+y] = tanh(r + b[y]);
        }
    }
}
"""

kernel_code_lstm = """
#if __OPENCL_VERSION__ <= CL_VERSION_1_1
#pragma OPENCL EXTENSION cl_khr_fp64: enable
#endif

__kernel __attribute__((reqd_work_group_size(WORK_ITEMS, 1, 1))) 
void run_dot(
    int iWx,
    int iWy,
    __global const FPTYPE* restrict inMat,
    __global const FPTYPE* restrict iW,
    __global FPTYPE* restrict outvW
) {
    int local_id = get_local_id(0);
    int local_size = get_local_size(0);
    int group_id = get_group_id(0);
    
    for(int x = 0; x < iWy; x += local_size)
    {
        FPTYPE vW = 0.0FSUF;
        for(int y = 0; y < iWx; ++y)
            vW += inMat[group_id*iWx + y] * iW[y*iWy + local_id + x];
        outvW[group_id*iWy + local_id + x] = vW;
    }
}

#define sigmoid(X) (1.0FSUF/(1.0FSUF + exp(-(X))))

__kernel __attribute__((reqd_work_group_size(WORK_ITEMS, 1, 1))) 
void run_lstm_layer(
    int inMatx,
    __global const FPTYPE* restrict outvW, 
    __global const FPTYPE* restrict lW, 
    __global const FPTYPE* restrict b, 
    __global const FPTYPE* restrict p,
    __global FPTYPE* restrict out
) {
    int local_id = get_global_id(0);
    //int local_size = get_local_size(0);
    
    FPTYPE state = 0.0FSUF;
    FPTYPE prev_state = 0.0FSUF;
    __local FPTYPE out_prev[WORK_ITEMS/4];
    __local FPTYPE sumW[WORK_ITEMS];
    
    if(local_id < 64)
        out_prev[local_id] = 0.0FSUF;
    barrier(CLK_LOCAL_MEM_FENCE);
    
    for(int i = 0; i < inMatx; ++i)
    {
        sumW[local_id] = 0.0FSUF;
        for(int y = 0; y < L_WX; ++y)
            sumW[local_id] += out_prev[y] * lW[y*L_WY + local_id];

        sumW[local_id] = outvW[i*L_WY + local_id] + sumW[local_id] + b[local_id];
        
        barrier(CLK_LOCAL_MEM_FENCE);
        if(local_id < 64)
        {
            prev_state = state;
            // Forget gate activation
            state = prev_state * sigmoid(sumW[2*64+local_id] + prev_state * p[1*L_WX + local_id]);
            // Update state with input
            state += tanh(sumW[0*64+local_id]) * sigmoid(sumW[1*64+local_id] + prev_state * p[0*L_WX + local_id]);
            // Output gate activation
            out_prev[local_id] = tanh(state) * sigmoid(sumW[3*64+local_id]  + state * p[2*L_WX + local_id]);
            out[i*L_WX + local_id] = out_prev[local_id];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }      
}
"""

kernel_code_softmax = """
#if __OPENCL_VERSION__ <= CL_VERSION_1_1
#pragma OPENCL EXTENSION cl_khr_fp64: enable
#endif

__kernel __attribute__((reqd_work_group_size(WORK_ITEMS, 1, 1)))
void run_layer(
    int inMatx,
    int Wx, 
    __global const FPTYPE* restrict inMat, 
    __global const FPTYPE* restrict W, 
    __global const FPTYPE* restrict b, 
    __global FPTYPE* restrict ret
){
    int id = get_global_id(0);
    if(id < inMatx)
    {
        FPTYPE inMatBuffer[IN_MAT_Y];
        for(int z = 0; z < IN_MAT_Y; ++z)
            inMatBuffer[z] = inMat[id*IN_MAT_Y+z];
        
        for(int y = 0; y < Wx; ++y)
        {
            FPTYPE r = 0.0FSUF;
            for(int z = 0; z < IN_MAT_Y; ++z)
                r += inMatBuffer[z] * W[y*IN_MAT_Y+z];
            ret[id*Wx+y] = r + b[y];
        }
    }
}

inline void parallel_sum(__local FPTYPE * restrict buffer)
{
    // Perform parallel reduction
    int local_index = get_local_id(0);
    for (int offset = get_local_size(0) / 2; offset > 0; offset = offset / 2) 
    {
        if (local_index < offset) 
            buffer[local_index] += buffer[local_index + offset];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}

inline void parallel_max(__local FPTYPE * restrict buffer)
{
    // Perform parallel reduction
    int local_index = get_local_id(0);
    for (int offset = get_local_size(0) / 2; offset > 0; offset = offset / 2) 
    {
        if (local_index < offset)
            if (buffer[local_index + offset] > buffer[local_index]) 
                buffer[local_index] = buffer[local_index + offset];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}

__kernel __attribute__((reqd_work_group_size(WORK_ITEMS_PAR, 1, 1)))       // WORK_ITEMS_PAR = 2^n
void run_softmax(
    int size,                         // 2^n+1
    __global FPTYPE * restrict inout
){
    int local_id = get_local_id(0);
    int local_size = get_local_size(0);
    int group_id = get_group_id(0);
    
    __local FPTYPE buffer[WORK_ITEMS_PAR];
    __local FPTYPE last_in;
    FPTYPE elem[ITER];
    FPTYPE max = 0.0FSUF;
    FPTYPE sum = 0.0FSUF;
    
    if(local_id == 0)
        last_in = inout[(group_id+1) * size - 1]; // access last size-1 element
    barrier(CLK_LOCAL_MEM_FENCE);
    max = last_in;
    
    for(int x = 0; x < size-1; x += local_size)
    {
        elem[x/local_size] = buffer[local_id] = inout[group_id * size + local_id + x];
        barrier(CLK_LOCAL_MEM_FENCE);
        parallel_max(buffer);
        max = max > buffer[0] ? max : buffer[0];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    for(int x = 0; x < size-1; x += local_size)
        elem[x/local_size] = exp(elem[x/local_size] - max); 

    if(local_id == 0)
        last_in = exp(last_in-max);

    for(int x = 0; x < size-1; x += local_size)
    {
        buffer[local_id] = elem[x/local_size];
        barrier(CLK_LOCAL_MEM_FENCE);
        parallel_sum(buffer);
        sum += buffer[0];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    sum += last_in;
    
    for(int x = 0; x < size-1; x += local_size)
        inout[group_id * size + local_id + x] = elem[x/local_size] / sum;
    if(local_id == 0)
        inout[(group_id+1) * size - 1] = last_in / sum;  
}
""" 
