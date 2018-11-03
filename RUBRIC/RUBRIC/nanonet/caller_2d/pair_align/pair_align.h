#ifndef PAIR_ALIGN_H
#define PAIR_ALIGN_H

#include <stdint.h>
#include <iostream>
#include <vector>
#include <boost/numeric/ublas/matrix.hpp>

namespace ublas = boost::numeric::ublas;

/// Namespace for pairwise alignment code.
namespace PairAlign {

/// Magic number representing log of zero (-INF).
static const int32_t ZERO_PROB_SCORE = -1000000000;

/// Helper function for the max of three values.
inline int32_t TripleMax(int32_t a, int32_t b, int32_t c) {
  return (a >= b) ? ((a >= c) ? a : c) : ((b >= c) ? b : c);
}

/// Helper function for the index of the max of three values.
inline int TripleMaxIndex(int32_t a, int32_t b, int32_t c) {
  return (a >= b) ? ((a >= c) ? 0 : 2) : ((b >= c) ? 1 : 2);
}

/// Helper struct for representing a position in an alignment.
struct AlignPos {
  int Pos1;
  int Pos2;
  /// Constructor.
  AlignPos(int p1 = 0, int p2 = 0) : Pos1(p1), Pos2(p2) {}
  /// Comparison operator. Sorts by first pos, then by second.
  bool operator<(const AlignPos& rhs) const {
    if (Pos1 == -1 || rhs.Pos1 == -1 || Pos1 == rhs.Pos1) {
      return Pos2 < rhs.Pos2;
    }
    return Pos1 < rhs.Pos1;
  }
};


/// Abstract baseclass for pairwise alignment.
class Aligner {
public:
  /// Destructor.
  virtual ~Aligner() {}

  /** Align two sequences.
   *  @param[in] sequence1 First sequence of states.
   *  @param[in] sequence2 Second sequence of states.
   *  @param[out] alignment Vector to hold the resulting alignment.
   *  @return The alignment score. This is not normalized.
   *
   *  The alignment object will contain one entry per alignment position. Any contents it
   *  had before the call will be lost. Each entry contains the indexes of the two sequence
   *  elements that align to that position. If one sequence has aligned to a gap at that
   *  position, the value for the other sequence will be -1.
   *
   *  Note that the two sequences must contain only values from 0 to n-1, where n is the
   *  size of the nxn substitution matrix.
   */
  virtual int32_t Align(const std::vector<int>& sequence1, const std::vector<int>& sequence2,
                        std::vector<AlignPos>& alignment) = 0;
};


} /* namespace PairAlign */


#endif /* PAIR_ALIGN_H */
