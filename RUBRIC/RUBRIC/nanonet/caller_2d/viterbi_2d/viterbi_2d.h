#ifndef VITERBI_2D_H
#define VITERBI_2D_H

#include <vector>
#include <string>
#include <stdint.h>
#include <boost/numeric/ublas/matrix.hpp>
#include <data_view.h>

namespace ublas = boost::numeric::ublas;

static const int8_t MOVE_DIAG  = 0;
static const int8_t MOVE_RIGHT = 1;
static const int8_t MOVE_UP    = 2;
static const int8_t MOVE_UNDEF = 3;
static const int32_t ZERO_PROB_SCORE = -1000000000;
static const double MIN_EMISSION_SCORE = -20.0;


inline double square(double x) {
  return x * x;
}

inline int32_t prob2score(double x) {
  if (x < 0.0000000001) return -2400;
  return int32_t(100.0 * log(x));
}


/// Helper class representing a node in the HMM.
struct Node {
  int32_t index1;                     ///< Index of event from first sequence.
  int32_t index2;                     ///< Index of event from second sequence.
  int32_t leftIndex;                  ///< Index of node to the left of this one.
  int32_t downIndex;                  ///< Index of node below this one.
  int32_t diagIndex;                  ///< Index of node diagonal to this one.
  std::vector<int16_t> statePointers; ///< Viterbi backtrace pointers.
  std::vector<int8_t> dirPointers;    ///< NW alignment backtrace pointers.

  /** Initialize node.
   *  @param[in] i Index of event from first sequence.
   *  @param[in] j index of event from second sequence.
   *  @param[in] left Index of node to the left of this one.
   *  @param[in] down Index of node below this one.
   *  @param[in] diag Index of node diagonal to this one.
   *  @param[in] states Number of states in the HMM.
   */
  void Init(int i, int j, int left, int down, int diag, int states) {
    index1 = i;
    index2 = j;
    leftIndex = left;
    downIndex = down;
    diagIndex = diag;
    statePointers.resize(states);
    dirPointers.resize(states);
  }
};


/** Helper class for emission scores.
 *
 *  This class provides normal level emissions and gamma distributed noise emissions.
 *  Note that other emission objects can be substituted by changing the Emission typedef
 *  immediately following this class definition.
 */
class DefaultEmission {
private:
  std::vector<double> levels;
  std::vector<double> noises;
  std::vector<double> logNoises;
  std::vector<double> stayWeights;
  std::vector<double> emWeights;
  std::vector<double> modelLevels;
  std::vector<double> modelNoises;
  std::vector<double> offsets;
  std::vector<double> levelScales;
  std::vector<double> noiseScales;
  std::vector<double> noiseShapes;
  int numEvents;
  int numStates;
  bool useNoise;

public:
  /** Constructor.
   *  @param[in] mdlLevels Model current levels.
   *  @param[in] mdlLevelSpreads Spreads of model current levels.
   *  @params[in] mdlNoises Model noise levels.
   *  @param[in] mdlNoiseSpreads Spreads of model noise levels.
   *  @param[in] useSd Flag to specify whether to use noise levels in the basecall.
   */
  DefaultEmission(const std::vector<double>& mdlLevels, const std::vector<double>& mdlLevelSpreads,
                  const std::vector<double>& mdlNoises, const std::vector<double>& mdlNoiseSpreads,
                  bool useSd);

  /** Assign events to the object with vectors.
   *  @param[in] means Event current levels.
   *  @param[in] stdvs Event noise levels.
   *  @param[in] stayWts Event weights for modifying stay probabilities.
   *  @param[in] emWts Event weights for modifying emission probabilities.
   */
  void SetEvents(const std::vector<double>& means, const std::vector<double>& stdvs,
                 const std::vector<double>& stayWts, const std::vector<double>& emWts);

  /// Set the number of events (for when SetEvents() will not be called.
  void SetNEvents(int n) {numEvents = n;}

  /// Returns the number of events.
  int NumEvents() const {return numEvents;}

  /// Returns the number of model states.
  int NumStates() const {return numStates;}

  /// Returns the model levels.
  const std::vector<double> GetModelLevels() const {return modelLevels;}

  /// Returns the stay weights.
  const std::vector<double> GetStayWeights() const {return stayWeights;}

  /// Returns the score for event i and state j.
  int32_t Score(int i, int j) const {
    double score = offsets[j] + levelScales[j] * square(levels[i] - modelLevels[j]);
    if (useNoise) score += (noiseShapes[j] - 1.0) * logNoises[i] - noiseScales[j] * noises[i];
    return int32_t(emWeights[i] * std::max(MIN_EMISSION_SCORE, score));
  }
};


typedef DefaultEmission Emission;
typedef std::vector<std::pair<int32_t, int32_t> > Alignment;


/// Worker class for performing 2D Viterbi basecall.
class Viterbi2D {
private:
  std::vector<Node> nodes;                         // All HMM nodes, in the order they should be processed.
  int32_t baseStay[3];                             // Stay scores for each direction.
  int32_t baseStep[3];                             // Step scores for each direction.
  int32_t baseSkip[3];                             // Skip scores for each direction.
  ublas::matrix<int32_t> emScore1;                 // Pre-computed emissions for sequence 1.
  ublas::matrix<int32_t> emScore2;                 // Pre-computed emissions for sequence 2.
  ublas::matrix<int32_t> viterbiScore;             // Viterbi scores. Length of sequence 1 by number of states.
  ublas::matrix<int32_t> lastScore;                // Viterbi scores for previous event from sequence 2.
  int numStates;                                   // Number of states in the HMM.
  int numNodes;                                    // Total number of nodes to be processed.
  int numEvents1;                                  // Number of events in sequence 1.
  int numEvents2;                                  // Number of events in sequence 2.

  void initNodes(const std::vector<int32_t>& bandStarts, const std::vector<int32_t>& bandEnds);
  void processNodes(const std::vector<double>& wts1, const std::vector<double>& wts2);
  void backTrace(Alignment& alignment, std::vector<int16_t>& states);

public:
  /** Constructor.
   *  @param[in] maxNodes The maximum number of nodes to support.
   *  @param[in] maxLen The maximum number of events to support for either sequence.
   *  @param[in] states The number of states in the HMM.
   *  @param[in] trans The six transition probabilities (stay1, step1, skip1, stay2, step2, skip2).
   */
  Viterbi2D(int maxNodes, int maxLen, int states, const std::vector<double>& trans);

  /** Perform the basecall with emission objects.
   *  @param[in] data1 Emission object for sequence 1.
   *  @param[in] data2 Emission object for sequence 2.
   *  @param[in] bandStarts For each event in sequence 2, the first candidate position in sequence 1.
   *  @param[in] bandEnds For each event in sequence 2, the last candidate position in sequence 1.
   *  @param[in] priors The prior scores for the "before alignment" node. All zeros means no prior.
   *  @param[out] alignment The final alignment of events.
   *  @param[out] states The final basecalled states.
   */
  void Call(const Emission& data1, const Emission& data2, const std::vector<int32_t>& bandStarts,
            const std::vector<int32_t>& bandEnds, const std::vector<int32_t>& priors,
            Alignment& alignment, std::vector<int16_t>& states);

  /** Perform the basecall with precomputed emissions.
   *  @param[in] data1 Precomputed emissions for sequence 1.
   *  @param[in] data2 Precomputed emissions for sequence 2.
   *  @param[in] stayWt1 Stay weights for sequence 1.
   *  @param[in] stayWt2 Stay weights for sequence 2.
   *  @param[in] bandStarts For each event in sequence 2, the first candidate position in sequence 1.
   *  @param[in] bandEnds For each event in sequence 2, the last candidate position in sequence 1.
   *  @param[in] priors The prior scores for the "before alignment" node. All zeros means no prior.
   *  @param[out] alignment The final alignment of events.
   *  @param[out] states The final basecalled states.
   */
  void Call(const MatView<float>& data1, const MatView<float>& data2,
            const VecView<double>& stayWt1, const VecView<double>& stayWt2,
            const std::vector<int32_t>& bandStarts, const std::vector<int32_t>& bandEnds,
            const std::vector<int32_t>& priors, Alignment& alignment, std::vector<int16_t>& states);
};


#endif /* VITERBI_2D */
