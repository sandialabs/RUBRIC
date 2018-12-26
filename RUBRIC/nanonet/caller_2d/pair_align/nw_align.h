#ifndef NW_ALIGN_H
#define NW_ALIGN_H

#include <pair_align.h>

namespace PairAlign {


/** Needleman-Wunsch implementation supporting gap-extension.
 *  Note that this implementation is optimized for speed, but
 *  is quadratic in memory. For aligning long sequences use
 *  the Myers-Miller implementation instead.
 */
class NWAlign : public Aligner {
private:
  const ublas::matrix<int32_t>& subMatrix;
  ublas::matrix<int32_t> diagScores;
  ublas::matrix<int32_t> upScores;
  ublas::matrix<int32_t> rightScores;
  int32_t startGapy, endGapy, openGapy, extendGapy;
  int32_t startGapx, endGapx, openGapx, extendGapx;

  void processNode(int i, int j, int32_t ogx, int32_t egx, int32_t ogy, int32_t egy, int32_t m) {
    // Find the best diagonal movement score.
    int32_t score = TripleMax(diagScores(i - 1, j - 1), rightScores(i - 1, j - 1), upScores(i - 1, j - 1));
    // Find the best upward movement score.
    int32_t upScore1 = diagScores(i - 1, j) + ogy;
    int32_t upScore2 = rightScores(i - 1, j) + ogy;
    int32_t upScore3 = upScores(i - 1, j) + egy;
    // Find the best right movement score.
    int32_t rightScore1 = diagScores(i, j - 1) + ogx;
    int32_t rightScore2 = upScores(i, j - 1) + ogx;
    int32_t rightScore3 = rightScores(i, j - 1) + egx;
    diagScores(i, j) = score + m;
    upScores(i, j) = TripleMax(upScore1, upScore2, upScore3);
    rightScores(i, j) = TripleMax(rightScore1, rightScore2, rightScore3);
  }

  void backtrace(std::vector<AlignPos>& alignment);

public:
  /** Constructor.
   *  @param[in] subMat Substitution matrix. Note this stores a reference. Beware of lifetime.
   *  @param[in] gaps Vector of gap penalties (length 8).
   *
   *  The 8 values in the gap penalty vector should be as follows:
   *    start_gap1  Penalty for aligning sequence 1 to a gap before sequence 2.
   *    end_gap1    Penalty for aligning sequence 1 to a gap after sequence 2.
   *    open_gap1   Penalty for aligning sequence 1 to a new gap within sequence 1.
   *    extend_gap1 Penalty for extending a gap within sequence 2.
   *    start_gap2  Penalty for aligning sequence 2 to a gap before sequence 1.
   *    end_gap2    Penalty for aligning sequence 2 to a gap after sequence 1.
   *    open_gap2   Penalty for aligning sequence 2 to a new gap within sequence 1.
   *    extend_gap2 Penalty for extending a gap within sequence 1.
   */
  NWAlign(const ublas::matrix<int32_t>& subMat, const std::vector<int>& gaps) : subMatrix(subMat) {
    startGapx = -gaps[0];
    endGapx = -gaps[1];
    openGapx = -gaps[2];
    extendGapx = -gaps[3];
    startGapy = -gaps[4];
    endGapy = -gaps[5];
    openGapy = -gaps[6];
    extendGapy = -gaps[7];
  }

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
  int32_t Align(const std::vector<int>& sequence1, const std::vector<int>& sequence2,
                std::vector<AlignPos>& alignment);
};

} /* namespace PairAlign */


#endif /* NW_ALIGN_H */
