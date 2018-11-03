#ifdef ENABLE_FP64
#if __OPENCL_VERSION__ <= CL_VERSION_1_1
#pragma OPENCL EXTENSION cl_khr_fp64: enable
#endif
#endif

#define MOVE_DIAG   0
#define MOVE_RIGHT  1
#define MOVE_UP     2
#define MOVE_UNDEF  3

#define ZERO_PROB_SCORE -1000000000


__kernel __attribute__((reqd_work_group_size(WORK_ITEMS, 1, 1)))
void ProcessNodes(
    int firstNode,
    int firstOffset,
    int wrapAround,
    int sliceSize,
    int index1First,
    int firstLeftValid,
    int lastDownValid,
    int firstDiagonalValid,
    int lastDiagonalValid,
    __global int* restrict viterbiScore,     // maxSliceSize * 2 * numStates
    __global int* restrict transitions,      // 9 (3 Stay, 3 Step, 3 Skip)
    __global int* restrict stayBuf,          // maxSliceSize * 3 * numStates
    __global short* restrict ptrs,           // maxSliceSize * 3 * numStates
    __global int* restrict emScore1,         // maxLen * numStates
    __global int* restrict emScore2,         // maxLen * numStates
#ifdef ENABLE_FP64
    __global double* restrict weights        // 3 * numNodes
#else
    __global int* restrict weights_stay      // 3 * numNodes
#endif
)
{
    int localId = get_local_id(0);
    int groupId = get_group_id(0);
    int slicePos = groupId / 3;
    int nodeIndex = firstNode + slicePos;
    int dir = groupId % 3;

    int pos = firstOffset + slicePos * 2;
    if (dir == 1) { --pos; }
    else if (dir == 2) { ++pos; }
    pos = pos % wrapAround;
    if (pos < 0) pos += wrapAround;
    viterbiScore += pos * NUM_STATES;

    stayBuf += NUM_STATES * groupId;
    ptrs += NUM_STATES * groupId;
#ifdef ENABLE_FP64
    int weight_stay = (int)(weights[3 * nodeIndex + dir] * transitions[dir]);
#else
    int weight_stay = weights_stay[3 * nodeIndex + dir];
#endif
    int step = transitions[3 + dir];
    int skip = transitions[6 + dir];

    // Fill in scores from previous nodes.
    if ((slicePos == 0 && ((dir == MOVE_RIGHT && firstLeftValid == 0) ||
                          (dir == MOVE_DIAG && firstDiagonalValid == 0))) ||
        (slicePos == sliceSize - 1 && ((dir == MOVE_UP && lastDownValid == 0) ||
                                      (dir == MOVE_DIAG && lastDiagonalValid == 0))))
    {
        for (int x = 0; x < NUM_STATES; x += WORK_ITEMS) {
            stayBuf[x + localId] = ZERO_PROB_SCORE;
        }
        return;
    }

    int index1 = index1First + slicePos;
    int index2 = index1First - firstOffset - slicePos;
    emScore1 += NUM_STATES * index1;
    emScore2 += NUM_STATES * index2;

    for (int x = 0; x < NUM_STATES; x += WORK_ITEMS) {
        // Add transitions scores.
        int state = x + localId;
        int score = viterbiScore[x + localId] + weight_stay;

        // Set pointers for stay movement. Scores are already stay scores.
        int ptr = state;

        // Find maxima for each direction.
        for (int from = 0; from < NUM_STATES; from += NUM_STATES/4) {

            // Check the step movement scores. Update as needed.
            int buf = viterbiScore[from + (state / 4)] + step;
            if (buf > score) {
                score = buf;
                ptr = from + (state / 4);
            }

            // Check the skip movement scores. Update as needed.
            #pragma unroll
            for (int y = 0; y < 4; ++y) {
                int fromState = from + (y * NUM_STATES / 16) + (state / 16);
                int buf = viterbiScore[fromState] + skip;
                if (buf > score) {
                    score = buf;
                    ptr = fromState;
                }
            }
        }

        // Apply emission scores, depending on direction
        if (dir < 2)  { score += emScore1[state]; }
        if (dir != 1) { score += emScore2[state]; }

        // Write result
        stayBuf[state] = score;
        ptrs[state] = ptr;
    }
}


__kernel  __attribute__((reqd_work_group_size(WORK_ITEMS, 1, 1)))
void PickBest(
    int firstNode,
    int firstOffset,
    int wrapAround,
    __global int* restrict stayBuf_tab,      // maxSliceSize * 3 * numStates
    __global short* restrict ptrs_tab,       // maxSliceSize * 3 * numStates
    __global short* restrict statePointers,  // numNodes * numStates
    __global char* restrict dirPointers,     // numNodes * numStates
    __global int* restrict viterbiScore      // maxSliceSize * 2 * numStates
)
{
    int localId = get_local_id(0);
    int groupId = get_group_id(0);
    int nodeIndex = firstNode + groupId;
    __global int *stayBuf = &stayBuf_tab[3 * NUM_STATES * groupId];
    __global short *ptrs = &ptrs_tab[3 * NUM_STATES * groupId];

    // Since firstOffset varies by +/-1 per slice we alternate between writing even and odd buffers
    int pos = (firstOffset + groupId * 2) % wrapAround;
    if (pos < 0) pos += wrapAround;

    // Pick the best of the three for each state.
    for (int j = 0; j < NUM_STATES; j += WORK_ITEMS) {
        int state = j + localId;
        char dir = MOVE_UP;
        int score0 = stayBuf[state];
        int score1 = stayBuf[NUM_STATES + state];
        int score = stayBuf[2*NUM_STATES + state];

        if (score0 > score1 && score0 > score) {
            dir = MOVE_DIAG;
            score = score0;
        } else if (score1 > score) {
            dir = MOVE_RIGHT;
            score = score1;
        }
        viterbiScore[pos * NUM_STATES + state] = score;
        statePointers[nodeIndex * NUM_STATES + state] = ptrs[dir * NUM_STATES + state];
        dirPointers[nodeIndex * NUM_STATES + state] = dir;
    }
};
