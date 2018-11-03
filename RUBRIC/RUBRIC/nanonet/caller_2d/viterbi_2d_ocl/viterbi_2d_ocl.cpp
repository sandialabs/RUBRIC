#define _USE_MATH_DEFINES
#include <cstdlib>
#include <csignal>
#include <cmath>
#include <utility>
#include <functional>
#include <algorithm>
#include <numeric>
#include <set>
#include <sys/types.h>
#include <sys/stat.h>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include "viterbi_2d_ocl.h"

#if defined(_MSC_VER) || defined(__MINGW32__)
static const double M_PI = 3.14159265358979323846;
#endif
static const double SQRT_2PI = sqrt(2.0 * M_PI);

#define CHECKERR(msg) if (err != CL_SUCCESS) { throw std::runtime_error(msg " (" + std::string(proxy_cl_.ocl_error_to_string(err)) + ")"); }
#define SETARG(kernel, index, arg) err = kernel.setArg(index, arg); CHECKERR(#arg)


DefaultEmission::DefaultEmission(const std::vector<double>& mdlLevels, const std::vector<double>& mdlLevelSpreads,
    const std::vector<double>& mdlNoises, const std::vector<double>& mdlNoiseSpreads,
    bool useSd) :
    modelLevels(mdlLevels),
    modelNoises(mdlNoises),
    offsets(mdlLevels.size()),
    levelScales(mdlLevels.size()),
    noiseScales(mdlLevels.size()),
    noiseShapes(mdlLevels.size()),
    useNoise(useSd) {
    for (size_t i = 0; i < modelLevels.size(); ++i) {
        offsets[i] = -log(mdlLevelSpreads[i] * SQRT_2PI);
        levelScales[i] = -0.5 / square(mdlLevelSpreads[i]);
        if (useNoise) {
            noiseScales[i] = square(mdlNoiseSpreads[i]) / fabs(mdlNoises[i]);
            noiseShapes[i] = fabs(mdlNoises[i]) / noiseScales[i];
            double noiseLogScale = log(noiseScales[i]);
            double noiseLogGammaShape = lgamma(noiseShapes[i]);
            offsets[i] -= noiseShapes[i] * noiseLogScale + noiseLogGammaShape;
            noiseScales[i] = 1.0 / noiseScales[i];
        }
    }
    numStates = int(mdlLevels.size());
}

void DefaultEmission::SetEvents(const std::vector<double>& means, const std::vector<double>& stdvs,
    const std::vector<double>& stayWts, const std::vector<double>& emWts) {
    levels = means;
    noises = stdvs;
    logNoises = stdvs;
    stayWeights = stayWts;
    emWeights = emWts;
    numEvents = int(means.size());
    for (int i = 0; i < numEvents; ++i) {
        emWeights[i] *= 100.0;
        if (useNoise) logNoises[i] = log(logNoises[i]);
    }
}


Viterbi2Docl::Viterbi2Docl(proxyCL &proxy_cl) :
    proxy_cl_(proxy_cl),
    enable_fp64_(false)
{}


void Viterbi2Docl::Call(const Emission& data1, const Emission& data2, const std::vector<int32_t>& bandStarts,
    const std::vector<int32_t>& bandEnds, const std::vector<int32_t>& priors, Alignment& alignment,
    std::vector<int16_t>& states) {
    numEvents1 = data1.NumEvents();
    numEvents2 = data2.NumEvents();
    initNodes(bandStarts, bandEnds);

    // Compute emission scores.
    for (int i = 0; i < numEvents1; ++i) {
        for (int j = 0; j < numStates; ++j) {
            emScore1(i, j) = data1.Score(i, j);
        }
    }
    for (int i = 0; i < numEvents2; ++i) {
        for (int j = 0; j < numStates; ++j) {
            emScore2(i, j) = data2.Score(i, j);
        }
    }
    processNodes(data1.GetStayWeights(), data2.GetStayWeights(), priors);
    backTrace(alignment, states);
}


void Viterbi2Docl::Call(const MatView<float>& data1, const MatView<float>& data2,
                        const VecView<double>& stayWt1, const VecView<double>& stayWt2,
                        const std::vector<int32_t>& bandStarts, const std::vector<int32_t>& bandEnds,
                        const std::vector<int32_t>& priors, Alignment& alignment, std::vector<int16_t>& states) {
    numEvents1 = int(data1.size1());
    numEvents2 = int(data2.size1());
    initNodes(bandStarts, bandEnds);

    // Compute emission scores and weights.
    std::vector<double> weights1(stayWt1.begin(), stayWt1.end());
    std::vector<double> weights2(stayWt2.begin(), stayWt2.end());
    for (int i = 0; i < numEvents1; ++i) {
        for (int j = 0; j < numStates; ++j) {
            double score = (data1(i, j) > 2e-9f) ? log(data1(i, j)) : -20.0;
            emScore1(i, j) = int32_t(score * 100.0);
        }
    }
    for (int i = 0; i < numEvents2; ++i) {
        for (int j = 0; j < numStates; ++j) {
            double score = (data2(i, j) > 2e-9f) ? log(data2(i, j)) : -20.0;
            emScore2(i, j) = int32_t(score * 100.0);
        }
    }
    processNodes(weights1, weights2, priors);
    backTrace(alignment, states);
}


void Viterbi2Docl::initNodes(const std::vector<int32_t>& bandStarts, const std::vector<int32_t>& bandEnds) {
    nodes.slices.clear();
    nodes.maxSliceSize = 0;
    numNodes = 0;

    HmmNodesData::NodeSlice slice;
    slice.firstNode = 0;
    slice.index1 = 0;
    slice.index2 = 0;
    slice.firstLeftValid = false;
    slice.lastDownValid = false;
    slice.firstDiagonalValid = true;  // The first slice doesn't actually have a valid diagonal node, but 
    slice.lastDiagonalValid = true;   // we put the priors where the scores for a diagonal node would be
    while (slice.index2 < numEvents2 && slice.index1 <= bandEnds[slice.index2]) {
        slice.numNodes = 0;
        // Step diagonally (down-right) through the bands until we hit the edge
        for (int x = slice.index1, y = slice.index2; y >= 0 && x >= bandStarts[y] && x <= bandEnds[y]; y--, x++) {
            slice.numNodes++;
            numNodes++;
        }
        if (nodes.slices.size() >= 1) {
            HmmNodesData::NodeSlice& prevSlice = nodes.slices.back();
            slice.lastDownValid = ((prevSlice.index1 + prevSlice.numNodes) == (slice.index1 + slice.numNodes));
            slice.firstDiagonalValid = false;
            slice.lastDiagonalValid = false;
        }
        if (nodes.slices.size() >= 2) {
            HmmNodesData::NodeSlice& prevSlice = nodes.slices[nodes.slices.size() - 2];
            int sliceLast = slice.index1 + slice.numNodes - 1;
            slice.firstDiagonalValid = (slice.index1 > prevSlice.index1) && (slice.index1 <= prevSlice.index1 + prevSlice.numNodes);
            slice.lastDiagonalValid = (sliceLast > prevSlice.index1) && (sliceLast <= prevSlice.index1 + prevSlice.numNodes);
        }

        nodes.maxSliceSize = std::max(nodes.maxSliceSize, slice.numNodes);
        nodes.slices.push_back(slice);

        slice.firstNode = numNodes;
        if ((slice.index2 + 1 < numEvents2) &&
            (slice.index1 >= bandStarts[slice.index2 + 1]) &&
            (slice.index1 <= bandEnds[slice.index2 + 1]))
        {
            slice.index2++; // step up
            slice.firstLeftValid = false;
        } else {
            slice.index1++; // step right
            slice.firstLeftValid = true;
        }
    }

    nodes.statePointers.resize(numNodes, numStates);
    nodes.dirPointers.resize(numNodes, numStates);
}


bool Viterbi2Docl::InitCL(const std::string& srcKernelDir, const std::string& binKernelDir, std::string &error, 
    bool enable_fp64, size_t num_states, size_t work_group_size /*= 0*/)
{
    numStates = num_states;
    vendor v = proxy_cl_.get_selected_vendor();
    std::string build_options((v == vendor::nvidia) ? "-I. -Werror -cl-nv-verbose" : "-I. -Werror");

    // This was set up and tested on nVidia K520 GPUs on EC2
    // Each EC2 instance has one GPU with 1024 work groups per 8 vCPUs.
    // For 5mer models with 1024 states, we can fill the entire work group,
    // but for 6mer it appears to be most efficient to use 256.
    // TODO: work group size could use some additional testing on 5mer models.
    // See OFAN-1134.
    size_t set_work_group_size = work_group_size;
    if (work_group_size == 0) { set_work_group_size = (numStates == 4096) ? 256 : numStates; };
    set_work_group_size = std::min(set_work_group_size, proxy_cl_.get_max_work_group_size());
    proxy_cl_.set_work_group_size(set_work_group_size);
    
    if (enable_fp64)
    {
        if (proxy_cl_.fp64_extension_support(error))
            enable_fp64_ = true;
        else if (proxy_cl_.double_fp_support(error))
            enable_fp64_ = true;
        else
            return false;
    }

    if (!proxy_cl_.create_command_queue(false, false, error))
        return false;

    build_options += " -D WORK_ITEMS=" + std::to_string(set_work_group_size) + " -D NUM_STATES=" + std::to_string(numStates);
    if (enable_fp64_)
        build_options += " -D ENABLE_FP64";

    // Here we try to load a previously compiled binary file. If that fails, load the source file
    // and try to generate a binary file for future use.
    std::string kernelSrcFilePath = srcKernelDir + "/viterbi_2d.cl";
    std::string kernelBinFilePath = binKernelDir + "/viterbi_2d_" + std::to_string(set_work_group_size)
        + "_" + std::to_string(numStates) + ".cl.bin";

    struct stat srcStat;
    struct stat binStat;
    bool loadBinary = (stat(kernelBinFilePath.c_str(), &binStat) == 0) &&
        ((stat(kernelSrcFilePath.c_str(), &srcStat) != 0) ||
         binStat.st_mtime > srcStat.st_mtime);

    if (loadBinary) {
        if (!proxy_cl_.load_kernel_from_binary_file(kernelBinFilePath, build_options, error)) {
            loadBinary = false;
        }
    }

    if (!loadBinary) {
        if (!proxy_cl_.load_kernel_from_source_file(kernelSrcFilePath, error)) {
            error = "Failed to load kernel from source: " + error;
            return false;
        }
    }
    if (!proxy_cl_.build_kernel(build_options, error)) {
        error = "Failed to build kernel: " + error;
        return false;
    }
    cl::Program &program = proxy_cl_.get_program();

    cl_int err;
    kernelProcessNodes = cl::Kernel(program, "ProcessNodes", &err);
    if (err != CL_SUCCESS) {
        error = "Kernel ProcessNodes() failed (" + std::string(proxy_cl_.ocl_error_to_string(err)) + ")";
        return false;
    }

    kernelPickBest = cl::Kernel(program, "PickBest", &err);
    if (err != CL_SUCCESS) {
        error = "Kernel PickBest() failed (" + std::string(proxy_cl_.ocl_error_to_string(err)) + ")";
        return false;
    }

    if (!loadBinary && !proxy_cl_.output_binary(kernelBinFilePath, build_options, error)) {
        std::cerr << "Failed to write binary file " << kernelBinFilePath << ", " << error << std::endl;
    }
    error.clear();
    return true;
}


void Viterbi2Docl::InitData(int len, int states, const std::vector<double>& trans) {
    emScore1.resize(len, states, false);
    emScore2.resize(len, states, false);
    viterbiScore.resize(states);

    transProbs.resize(9);
    double stepMod = 0.25;
    double skipMod = 0.0625;
    transProbs[0] = prob2score(trans[0] * trans[3]);
    transProbs[1] = prob2score(trans[0]);
    transProbs[2] = prob2score(trans[3]);
    transProbs[3] = prob2score(trans[1] * trans[4] * stepMod);
    transProbs[4] = prob2score(trans[1] * trans[5] * stepMod);
    transProbs[5] = prob2score(trans[4] * trans[2] * stepMod);
    transProbs[6] = prob2score(trans[2] * trans[5] * skipMod);
    transProbs[7] = prob2score(trans[2] * square(trans[5]) * skipMod);
    transProbs[8] = prob2score(trans[5] * square(trans[2]) * skipMod);
}


void Viterbi2Docl::processNodes(const std::vector<double>& weights1, const std::vector<double>& weights2,
    const std::vector<int32_t>& priors)
{
    bool enable_profiling = proxy_cl_.profiling_enabled();
    size_t set_work_group_size = proxy_cl_.get_work_group_size();
    cl::Context &context = proxy_cl_.get_context();
    cl::CommandQueue &queue = proxy_cl_.get_command_queue();
    const int32_t wrapAround = nodes.maxSliceSize * 2;
    const int process_nodes_num_worgroups = 3;
    cl_int err;

    // Allocate and fill OpenCL buffers
    const cl_mem_flags memFlagsInput = CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR | CL_MEM_HOST_NO_ACCESS;
    const cl_mem_flags memFlagsOutput = CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY;
    const cl_mem_flags memFlagsInOut = CL_MEM_READ_WRITE;
    const cl_mem_flags memFlagsTemp = CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS;

    cl::Buffer cl_viterbiScore = cl::Buffer(context, memFlagsInOut,
        viterbiScore.size() * wrapAround * sizeof(cl_int), nullptr, &err);
    CHECKERR("cl_viterbiScore");
    err = queue.enqueueWriteBuffer(cl_viterbiScore, CL_FALSE, 0, sizeof(int32_t) * priors.size(), priors.data());
    CHECKERR("cl_viterbiScore");

    cl::Buffer cl_stayBuf = cl::Buffer(context, memFlagsTemp,
        3 * numStates * nodes.maxSliceSize * sizeof(cl_int), nullptr, &err);
    CHECKERR("cl_stayBuf");

    cl::Buffer cl_ptrs = cl::Buffer(context, memFlagsTemp,
        3 * numStates * nodes.maxSliceSize * sizeof(cl_short), nullptr, &err);
    CHECKERR("cl_ptrs");

    cl::Buffer cl_weights;
    // Initialise the node weights
    double minFactor = (double)transProbs[3] / (double)transProbs[0];
    if(enable_fp64_) {
        std::vector<double> weights(3 * numNodes);
        for (const HmmNodesData::NodeSlice& slice : nodes.slices) {
            for (int i = 0; i < slice.numNodes; ++i) {
                int nodeIndex = i + slice.firstNode;
                double weight1 = weights1[slice.index1 + i];
                double weight2 = weights2[slice.index2 - i];
                weights[3 * nodeIndex] = std::max(minFactor, 0.5 * (weight1 + weight2));
                weights[3 * nodeIndex + 1] = std::max(minFactor, weight1);
                weights[3 * nodeIndex + 2] = std::max(minFactor, weight2);
            }
        }
        cl_weights = cl::Buffer(context, memFlagsInput, weights.size() * sizeof(double), weights.data(), &err);
    } else {
        std::vector<int32_t> weightsStay(3 * numNodes);
        for (const HmmNodesData::NodeSlice& slice : nodes.slices) {
            for (int i = 0; i < slice.numNodes; ++i) {
                int nodeIndex = i + slice.firstNode;
                double weight1 = weights1[slice.index1 + i];
                double weight2 = weights2[slice.index2 - i];
                weightsStay[3 * nodeIndex] = int32_t(transProbs[0] * std::max(minFactor, 0.5 * (weight1 + weight2)));
                weightsStay[3 * nodeIndex + 1] = int32_t(transProbs[1] * std::max(minFactor, weight1));
                weightsStay[3 * nodeIndex + 2] = int32_t(transProbs[2] * std::max(minFactor, weight2));
            }
        }
        cl_weights = cl::Buffer(context, memFlagsInput, weightsStay.size() * sizeof(cl_int), weightsStay.data(), &err);
    }
    CHECKERR("cl_weights");

    int32_t transInt[9];
    for (int i = 0; i < 9; ++i) { transInt[i] = int32_t(transProbs[i]); }
    cl::Buffer cl_transitions = cl::Buffer(context, memFlagsInput,
        9 * sizeof(cl_int), transInt, &err);
    CHECKERR("cl_transitions");

    cl::Buffer cl_emScore1 = cl::Buffer(context, memFlagsInput,
        emScore1.size1() * emScore1.size2() * sizeof(cl_int), &emScore1.data()[0], &err);
    CHECKERR("cl_emScore1");

    cl::Buffer cl_emScore2 = cl::Buffer(context, memFlagsInput,
        emScore2.size1() * emScore2.size2() * sizeof(cl_int), &emScore2.data()[0], &err);
    CHECKERR("cl_emScore2");

    cl::Buffer cl_statePointers = cl::Buffer(context, memFlagsOutput,
        nodes.statePointers.size1() * nodes.statePointers.size2() * sizeof(cl_short), nullptr, &err);
    CHECKERR("cl_statePointers");

    cl::Buffer cl_dirPointers = cl::Buffer(context, memFlagsOutput,
        nodes.dirPointers.size1() * nodes.dirPointers.size2() * sizeof(char), nullptr, &err);
    CHECKERR("cl_dirPointers");

    cl::Event ev_enqueue_end, ev_state_pointers_end, ev_dir_pointers_end, ev_viterbi_score_end;

    // Set kernel buffer args and constants
    SETARG(kernelProcessNodes, 2, wrapAround);
    SETARG(kernelProcessNodes, 9, cl_viterbiScore);
    SETARG(kernelProcessNodes, 10, cl_transitions);
    SETARG(kernelProcessNodes, 11, cl_stayBuf);
    SETARG(kernelProcessNodes, 12, cl_ptrs);
    SETARG(kernelProcessNodes, 13, cl_emScore1);
    SETARG(kernelProcessNodes, 14, cl_emScore2);
    SETARG(kernelProcessNodes, 15, cl_weights);

    SETARG(kernelPickBest, 2, wrapAround);
    SETARG(kernelPickBest, 3, cl_stayBuf);
    SETARG(kernelPickBest, 4, cl_ptrs);
    SETARG(kernelPickBest, 5, cl_statePointers);
    SETARG(kernelPickBest, 6, cl_dirPointers);
    SETARG(kernelPickBest, 7, cl_viterbiScore);

    // Loop over slices, enqueue ProcessNodes and PickBest kernels for each slice
    for (const HmmNodesData::NodeSlice& slice : nodes.slices) {
        SETARG(kernelProcessNodes, 0, slice.firstNode);
        SETARG(kernelProcessNodes, 1, slice.index1 - slice.index2);
        SETARG(kernelProcessNodes, 3, slice.numNodes);
        SETARG(kernelProcessNodes, 4, slice.index1);
        SETARG(kernelProcessNodes, 5, int(slice.firstLeftValid));
        SETARG(kernelProcessNodes, 6, int(slice.lastDownValid));
        SETARG(kernelProcessNodes, 7, int(slice.firstDiagonalValid));
        SETARG(kernelProcessNodes, 8, int(slice.lastDiagonalValid));

        err = queue.enqueueNDRangeKernel(kernelProcessNodes, cl::NullRange,
            cl::NDRange(slice.numNodes * process_nodes_num_worgroups * set_work_group_size, 1),
            cl::NDRange(set_work_group_size, 1),
            0, enable_profiling ? &ev_enqueue_end : nullptr);
        CHECKERR("Enqueue kernelProcessNodes failed");

        SETARG(kernelPickBest, 0, slice.firstNode);
        SETARG(kernelPickBest, 1, slice.index1 - slice.index2);

        err = queue.enqueueNDRangeKernel(kernelPickBest, cl::NullRange,
            cl::NDRange(slice.numNodes * set_work_group_size, 1),
            cl::NDRange(set_work_group_size, 1),
            0, enable_profiling ? &ev_enqueue_end : nullptr);
        CHECKERR("Enqueue kernelPickBest failed");
	}

    // Read results
	err = queue.enqueueReadBuffer(cl_statePointers, CL_FALSE, 0,
	    nodes.statePointers.size1()*nodes.statePointers.size2()*sizeof(nodes.statePointers.data()[0]),
	    &nodes.statePointers.data()[0], 0, enable_profiling ? &ev_state_pointers_end : nullptr);
    CHECKERR("enqueueReadBuffer cl_statePointers failed");

	err = queue.enqueueReadBuffer(cl_dirPointers, CL_FALSE, 0,
	    nodes.dirPointers.size1()*nodes.dirPointers.size2()*sizeof(nodes.dirPointers.data()[0]),
	    &nodes.dirPointers.data()[0], 0, enable_profiling ? &ev_dir_pointers_end : nullptr);
    CHECKERR("enqueueReadBuffer cl_dirPointers failed");

    const HmmNodesData::NodeSlice& lastSlice = nodes.slices.back();
    int pos = lastSlice.index1 - lastSlice.index2;
    pos = pos % wrapAround;
    if (pos < 0) pos += wrapAround;

	err = queue.enqueueReadBuffer(cl_viterbiScore, CL_TRUE,
	    pos * viterbiScore.size() * sizeof(viterbiScore.data()[0]),
	    viterbiScore.size() * sizeof(viterbiScore.data()[0]),
	    viterbiScore.data(), 0, enable_profiling ? &ev_viterbi_score_end : nullptr);
    CHECKERR("enqueueReadBuffer cl_viterbiScore failed");
}


void Viterbi2Docl::backTrace(Alignment& alignment, std::vector<int16_t>& states) {
    alignment.clear();
    states.clear();
    int16_t state = int16_t(std::max_element(viterbiScore.begin(), viterbiScore.end()) - viterbiScore.begin());

    auto slice = nodes.slices.end() - 1;
    int slicePos = 0;
    do {
        int pos = slice->firstNode + slicePos;
        int32_t currSliceIndex1 = slice->index1;
        int16_t prevState = nodes.statePointers(pos, state);
        int8_t dir = nodes.dirPointers(pos, state);
        assert(slicePos >= 0);
        assert(slicePos < slice->numNodes);
        states.push_back(state);
        switch (dir) {
        case MOVE_DIAG:
            alignment.push_back(std::make_pair(slice->index1 + slicePos, slice->index2 - slicePos));
            slice -= 2;
            slicePos += currSliceIndex1 - slice->index1 - 1;
            break;
        case MOVE_RIGHT:
            alignment.push_back(std::make_pair(slice->index1 + slicePos, -1));
            slice -= 1;
            slicePos += currSliceIndex1 - slice->index1 - 1;
            break;
        case MOVE_UP:
            alignment.push_back(std::make_pair(-1, slice->index2 - slicePos));
            slice -= 1;
            slicePos += currSliceIndex1 - slice->index1;
            break;
        default:
            throw std::runtime_error("Error in backtrace. Unrecognised movement value.");
        }
        state = prevState;
    } while (slice != nodes.slices.begin());
    states.push_back(state);
    alignment.push_back(std::make_pair(slice->index1, slice->index2));

    reverse(alignment.begin(), alignment.end());
    reverse(states.begin(), states.end());
}
