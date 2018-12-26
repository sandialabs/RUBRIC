#include <cmath>
#include <utility>
#include <functional>
#include <algorithm>
#include <numeric>
#include <iostream>
#include <boost/math/special_functions/gamma.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <emmintrin.h>
#include "viterbi_2d.h"

using namespace std;
using ublas::matrix;
using ublas::matrix_row;
using ublas::matrix_column;

#if defined(_MSC_VER)
  static const double M_PI = 3.14159265358979323846;
#endif
static const double SQRT_2PI = sqrt(2.0 * M_PI);


DefaultEmission::DefaultEmission(const vector<double>& mdlLevels, const vector<double>& mdlLevelSpreads,
                                 const vector<double>& mdlNoises, const vector<double>& mdlNoiseSpreads,
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
      noiseScales[i] = square(mdlNoiseSpreads[i]) / abs(mdlNoises[i]);
      noiseShapes[i] = abs(mdlNoises[i]) / noiseScales[i];
      double noiseLogScale = log(noiseScales[i]);
      double noiseLogGammaShape = boost::math::lgamma(noiseShapes[i]);
      offsets[i] -= noiseShapes[i] * noiseLogScale + noiseLogGammaShape;
      noiseScales[i] = 1.0 / noiseScales[i];
    }
  }
  numStates = int(mdlLevels.size());
}


void DefaultEmission::SetEvents(const vector<double>& means, const vector<double>& stdvs,
                                const vector<double>& stayWts, const vector<double>& emWts) {
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


Viterbi2D::Viterbi2D(int maxNodes, int maxLen, int states, const vector<double>& trans) :
  emScore1(maxLen, states),
  emScore2(maxLen, states),
  viterbiScore(maxLen + 1, states),
  lastScore(maxLen + 1, states),
  numStates(states) {
  double stepMod = 0.25;
  double skipMod = 0.0625;
  baseStay[0] = prob2score(trans[0] * trans[3]);
  baseStep[0] = prob2score(trans[1] * trans[4] * stepMod);
  baseSkip[0] = prob2score(trans[2] * trans[5] * skipMod);
  baseStay[1] = prob2score(trans[0]);
  baseStep[1] = prob2score(trans[1] * trans[5] * stepMod);
  baseSkip[1] = prob2score(trans[2] * square(trans[5]) * skipMod);
  baseStay[2] = prob2score(trans[3]);
  baseStep[2] = prob2score(trans[4] * trans[2] * stepMod);
  baseSkip[2] = prob2score(trans[5] * square(trans[2]) * skipMod);
}


void Viterbi2D::Call(const Emission& data1, const Emission& data2, const vector<int32_t>& bandStarts,
                     const vector<int32_t>& bandEnds, const vector<int32_t>& priors, Alignment& alignment,
                     vector<int16_t>& states) {
  numEvents1 = data1.NumEvents();
  numEvents2 = data2.NumEvents();
  matrix_row<matrix<int32_t> > row(lastScore, 0);
  copy(priors.begin(), priors.end(), row.begin());
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
  const vector<double>& weights1 = data1.GetStayWeights();
  const vector<double>& weights2 = data2.GetStayWeights();
  processNodes(weights1, weights2);
  backTrace(alignment, states);
}


void Viterbi2D::Call(const MatView<float>& data1, const MatView<float>& data2,
                     const VecView<double>& stayWt1, const VecView<double>& stayWt2,
                     const vector<int32_t>& bandStarts, const vector<int32_t>& bandEnds,
                     const vector<int32_t>& priors, Alignment& alignment, vector<int16_t>& states) {
  numEvents1 = int(data1.size1());
  numEvents2 = int(data2.size1());
  matrix_row<matrix<int32_t> > row(lastScore, 0);
  copy(priors.begin(), priors.end(), row.begin());
  initNodes(bandStarts, bandEnds);

  // Compute emission scores and weights.
  vector<double> weights1(stayWt1.begin(), stayWt1.end());
  vector<double> weights2(stayWt2.begin(), stayWt2.end());
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
  processNodes(weights1, weights2);
  backTrace(alignment, states);
}


void Viterbi2D::initNodes(const vector<int32_t>& bandStarts, const vector<int32_t>& bandEnds) {
  numNodes = 1;
  for (int i = 0; i < numEvents2; ++i) {
    numNodes += bandEnds[i] - bandStarts[i] + 1;
  }
  nodes.clear();
  nodes.resize(numNodes);
  nodes[0].Init(-1, -1, -1, -1, -1, numStates);
  fill(nodes[0].dirPointers.begin(), nodes[0].dirPointers.end(), MOVE_UNDEF);
  fill(nodes[0].statePointers.begin(), nodes[0].statePointers.end(), int16_t(0));
  int node = 1;
  for (int i = 0; i < numEvents2; ++i) {
    for (int j = bandStarts[i]; j <= bandEnds[i]; ++j, ++node) {
      int left, down, diag;
      if (j == bandStarts[i]) left = -1;
      else left = node - 1;
      if (i == 0) down = -1;
      else if (j < bandStarts[i - 1] || j > bandEnds[i - 1]) down = -1;
      else down = node - (bandEnds[i - 1] - bandStarts[i] + 1);
      if (node == 1) diag = 0;
      else if (i == 0) diag = -1;
      else if (j <= bandStarts[i - 1] || j > bandEnds[i - 1] + 1) diag = -1;
      else diag = node - (bandEnds[i - 1] - bandStarts[i] + 2);
      nodes[node].Init(j, i, left, down, diag, numStates);
    }
  }
}


void Viterbi2D::processNodes(const vector<double>& weights1, const vector<double>& weights2) {
  vector<vector<int32_t> > stayBuf(3);
  vector<vector<int32_t> > stepBuf(3);
  vector<vector<int32_t> > skipBuf(3);
  for (int i = 0; i < 3; ++i) {
    stayBuf[i].resize(numStates);
    stepBuf[i].resize(numStates * 4);
    skipBuf[i].resize(numStates * 16);
  }

  int currentRow = 0;
  for (int nodeIndex = 1; nodeIndex < numNodes; ++nodeIndex) {
    Node& node = nodes[nodeIndex];
    int index1 = node.index1;
    int index2 = node.index2;
    int pos = index1 + 1;
    int indexes[3];
    indexes[0] = node.diagIndex;
    indexes[1] = node.leftIndex;
    indexes[2] = node.downIndex;

    // Swap score with lastScore if necessary.
    if (currentRow != index2) {
      currentRow = index2;
      lastScore.swap(viterbiScore);
    }

    // Fill in scores from previous nodes and add transitions scores.
    double weights[3];
    double minFactor = double(baseStep[0]) / double(baseStay[0]);
    weights[0] = max(minFactor, 0.5 * (weights1[index1] + weights2[index2]));
    weights[1] = max(minFactor, weights1[index1]);
    weights[2] = max(minFactor, weights2[index2]);

    for (int dir = 0; dir < 3; ++dir) {
      int32_t wgtStay = int32_t(weights[dir] * baseStay[dir]);
      if (indexes[dir] == -1) {
        fill(stayBuf[dir].begin(), stayBuf[dir].end(), ZERO_PROB_SCORE + wgtStay);
      }
      else {
        matrix_row<matrix<int32_t> > lastRow(
          (dir == 1) ? viterbiScore : lastScore,
          (dir == 2) ? pos : pos - 1);
        __m128i sseWgtStay = _mm_set1_epi32(wgtStay);
        for (size_t x = 0; x < numStates; x += 4) {
          __m128i sseStayBuf = _mm_loadu_si128((const __m128i*)&lastRow[x]);
          _mm_storeu_si128((__m128i*)&stayBuf[dir][x], _mm_add_epi32(sseStayBuf, sseWgtStay));
        }
      }
      __m128i sseStep = _mm_set1_epi32(baseStep[dir] - wgtStay);
      __m128i sseSkip = _mm_set1_epi32(baseSkip[dir] - wgtStay);
      vector<int32_t>::const_iterator p0 = stayBuf[dir].begin();
      vector<int32_t>::iterator p1 = stepBuf[dir].begin();
      vector<int32_t>::iterator p2 = skipBuf[dir].begin();
      for (; p0 < stayBuf[dir].end(); ++p0) {
        __m128i ssePo = _mm_set1_epi32(*p0);
        _mm_storeu_si128((__m128i*)&*p1, _mm_add_epi32(ssePo, sseStep));
        p1 += 4;
        for (int j = 0; j < 16; j+=4, p2+=4) 
          _mm_storeu_si128((__m128i*)&*p2, _mm_add_epi32(ssePo, sseSkip));
      }
    }

    // Find maximums for each direction.
    // for the future reference ptrs can be int16_t
    // but is int32_t for sse simplicity
    vector<vector<int32_t> > ptrs(3);
    for (int dir = 0; dir < 3; ++dir) {
      // Set pointers for stay movement. Scores are already stay scores.
      ptrs[dir].resize(numStates);
      vector<int32_t>::iterator it = ptrs[dir].begin();
      for (int i = 0; i < numStates; ++i, ++it) *it = int16_t(i);

      // Check the step movement scores. Update as needed.
      vector<int32_t>::iterator p0 = stayBuf[dir].begin();
      vector<int32_t>::iterator p1 = stepBuf[dir].begin();
      it = ptrs[dir].begin();
      for (int from = 0; from < numStates; ) {
        p0 = stayBuf[dir].begin();
        it = ptrs[dir].begin();
        for (int a = 0; a < numStates / 4; ++a, ++from) {
          __m128i ssep1 = _mm_loadu_si128((const __m128i*)&*p1);
          __m128i ssep0 = _mm_loadu_si128((const __m128i*)&*p0);
          __m128i ssecmp = _mm_cmpgt_epi32(ssep1, ssep0);
          _mm_storeu_si128((__m128i*)&*p0, _mm_or_si128(_mm_and_si128(ssecmp, ssep1), _mm_andnot_si128(ssecmp, ssep0)));
          __m128i ssefrom = _mm_set1_epi32(from);
          __m128i sseit = _mm_loadu_si128((const __m128i*)&*it);
          _mm_storeu_si128((__m128i*)&*it, _mm_or_si128(_mm_and_si128(ssecmp, ssefrom), _mm_andnot_si128(ssecmp, sseit)));
          p0 += 4;
          p1 += 4;
          it += 4;
        }
      }

      // Check the skip movement scores. Update as needed.
      vector<int32_t>::iterator p2 = skipBuf[dir].begin();
      p0 = stayBuf[dir].begin();
      it = ptrs[dir].begin();
      for (int from = 0; from < numStates; ) {
        p0 = stayBuf[dir].begin();
        it = ptrs[dir].begin();
        for (int a = 0; a < numStates / 16; ++a, ++from) {
          __m128i ssefrom = _mm_set1_epi32(from);
          for (int i = 0; i < 16; i+=4, p0+=4, p2+=4, it+=4) {
            __m128i ssep2 = _mm_loadu_si128((const __m128i*)&*p2);
            __m128i ssep0 = _mm_loadu_si128((const __m128i*)&*p0);
            __m128i ssecmp = _mm_cmpgt_epi32(ssep2, ssep0);
            _mm_storeu_si128((__m128i*)&*p0, _mm_or_si128(_mm_and_si128(ssecmp, ssep2), _mm_andnot_si128(ssecmp, ssep0)));
            __m128i sseit = _mm_loadu_si128((const __m128i*)&*it);
            _mm_storeu_si128((__m128i*)&*it, _mm_or_si128(_mm_and_si128(ssecmp, ssefrom), _mm_andnot_si128(ssecmp, sseit)));
          }
        }
      }
    }

    // Add emission values.
    matrix_row<matrix<int32_t> > em1(emScore1, index1);
    matrix_row<matrix<int32_t> > em2(emScore2, index2);
    matrix_row<matrix<int32_t> >::const_iterator emIt1 = em1.begin();
    matrix_row<matrix<int32_t> >::const_iterator emIt2 = em2.begin();
    vector<int32_t>::iterator p0 = stayBuf[0].begin();
    vector<int32_t>::iterator p1 = stayBuf[1].begin();
    vector<int32_t>::iterator p2 = stayBuf[2].begin();
    for (; p0 < stayBuf[0].end(); p0+=4, p1+=4, p2+=4, emIt1+=4, emIt2+=4) {
      __m128i sseem1 = _mm_loadu_si128((const __m128i*)&*emIt1);
      __m128i sseem2 = _mm_loadu_si128((const __m128i*)&*emIt2);
      __m128i ssep0 = _mm_loadu_si128((const __m128i*)&*p0);
      _mm_storeu_si128((__m128i*)&*p0, _mm_add_epi32(ssep0, _mm_add_epi32(sseem1, sseem2)));
      __m128i ssep1 = _mm_loadu_si128((const __m128i*)&*p1);
      _mm_storeu_si128((__m128i*)&*p1, _mm_add_epi32(ssep1, sseem1));
      __m128i ssep2 = _mm_loadu_si128((const __m128i*)&*p2);
      _mm_storeu_si128((__m128i*)&*p2, _mm_add_epi32(ssep2, sseem2));
    }

    // Pick the best of the three for each state.
    for (int j = 0; j < numStates; ++j) {
      if (stayBuf[0][j] > stayBuf[1][j] && stayBuf[0][j] > stayBuf[2][j]) {
        viterbiScore(pos, j) = stayBuf[0][j];
        node.statePointers[j] = ptrs[0][j];
        node.dirPointers[j] = MOVE_DIAG;
      }
      else if (stayBuf[1][j] > stayBuf[2][j]) {
        viterbiScore(pos, j) = stayBuf[1][j];
        node.statePointers[j] = ptrs[1][j];
        node.dirPointers[j] = MOVE_RIGHT;
      }
      else {
        viterbiScore(pos, j) = stayBuf[2][j];
        node.statePointers[j] = ptrs[2][j];
        node.dirPointers[j] = MOVE_UP;
      }
    }
  }
}


void Viterbi2D::backTrace(Alignment& alignment, vector<int16_t>& states) {
  alignment.clear(); states.clear();
  int lastIndex1 = nodes.back().index1;
  matrix_row<matrix<int32_t> > prevScore(viterbiScore, lastIndex1 + 1);
  int16_t state = int16_t(max_element(prevScore.begin(), prevScore.end()) - prevScore.begin());
  int pos = int(nodes.size() - 1);
  while (pos > 0) {
    int16_t prevState = nodes[pos].statePointers[state];
    int8_t dir = nodes[pos].dirPointers[state];
    states.push_back(state);
    switch(dir) {
    case MOVE_DIAG:
      alignment.push_back(make_pair(nodes[pos].index1, nodes[pos].index2));
      pos = nodes[pos].diagIndex;
      break;
    case MOVE_RIGHT:
      alignment.push_back(make_pair(nodes[pos].index1, -1));
      pos = nodes[pos].leftIndex;
      break;
    case MOVE_UP:
      alignment.push_back(make_pair(-1, nodes[pos].index2));
      pos = nodes[pos].downIndex;
      break;
    default:
      throw runtime_error("Error in backtrace. Unrecognised movement value.");
    }
    state = prevState;
  }
  reverse(alignment.begin(), alignment.end());
  reverse(states.begin(), states.end());
}

