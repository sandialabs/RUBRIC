#ifndef MM_ALIGN_H
#define MM_ALIGN_H

#include <pair_align.h>

namespace PairAlign {

/// Helper class representing a view of a range of a vector.
template <class T>
class View {
private:
  const std::vector<T> *data;
  size_t start, size;
  int dir;

  void check(size_t a, size_t b) {
    if (start < a || start > b) {
      throw std::runtime_error("Error: View start is out of range.");
    }
    size_t end = start + size_t((int(size) - 1) * dir);
    if (end < a || end > b) {
      throw std::runtime_error("Error: View end is out of range.");
    }
    if (dir != 1 && dir != -1) {
      throw std::runtime_error("Error: Stride value makes no sense.");
    }
  }

public:
  /// Default constructor.
  View() {}

  /** Construct a subview.
   *  @param[in] rhs View object to view contents of.
   *  @param[in] begin Start position for new view.
   *  @param[in] len Length of new view.
   *  @param[in] dir Either 1 or -1. Indicates direction of view.
   */
  View(const View& rhs, int begin, int len, int d) :
    data(rhs.data),
    start(rhs.start + begin * rhs.dir),
    size(len),
    dir(d * rhs.dir) {
    check(rhs.start, rhs.start + (rhs.size - 1) * rhs.dir);
  }

  /** Construct a view of a std::vector.
   *  @param[in] x Vector to be viewed.
   *  @param[in] begin Start position of view.
   *  @param[in] len Length of view.
   *  @param[in] dir Either 1 or -1. Indicates direction of view.
   */
  View(const std::vector<T>& x, int begin = 0, int len = 0, int d = 1) :
    data(&x),
    start(begin),
    size(len == 0 ? x.size() : len),
    dir(d) {
    check(0, x.size() - 1);
  }

  /// Indexing operator.
  const T& operator[](int n) const {return (dir == 1) ? (*data)[start + n] : (*data)[start - n];}

  /// Returns the lenght of the view.
  size_t Size() const {return size;}
};


/** Myers-Miller implementation supporting gap-extension.
 *  Note that this is approximately 2x slower than the
 *  Needleman-Wunsch implementation, but only requires
 *  linear memory instead of quadratic.
 */
class MMAlign : public Aligner {
private:
  const ublas::matrix<int32_t>& subMatrix;
  std::vector<int32_t> M, lastM, buffM;
  std::vector<int32_t> Iy, lastIy, buffIy;
  std::vector<int32_t> Ix, lastIx;
  std::vector<AlignPos> matches;
  View<int> seq1;
  View<int> seq2;
  int32_t startGapx, endGapx, openGapx, extendGapx;
  int32_t startGapy, endGapy, openGapy, extendGapy;

  int32_t processBlock(int xpos1, int xpos2, int ypos1, int ypos2,
                       int32_t m1, int32_t iy1, int32_t m2, int32_t iy2);
  void processUp(int xpos1, int xpos2, int ypos1, int ypos2,
                 int32_t m, int32_t iy);
  void processDown(int xpos1, int xpos2, int ypos1, int ypos2,
                   int32_t m, int32_t iy);
  void hmm(const View<int>& view1, const View<int>& view2, int lenx, int leny,
           int32_t m, int32_t iy, int32_t gx1, int32_t hx1, int32_t gx2,
           int32_t hx2, int32_t gy1, int32_t hy1, int32_t gy2, int32_t hy2);
  void makeAlignment(std::vector<AlignPos>& alignment);

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
  MMAlign(const ublas::matrix<int32_t>& subMat, const std::vector<int>& gaps) : subMatrix(subMat) {
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
                std::vector<AlignPos>& alignment) {
    seq1 = View<int>(sequence1);
    seq2 = View<int>(sequence2);
    int len1 = int(sequence1.size());
    int len2 = int(sequence2.size());
    M.clear();
    lastM.clear();
    buffM.clear();
    Ix.clear();
    lastIx.clear();
    Iy.clear();
    lastIy.clear();
    buffIy.clear();
    matches.clear();
    M.resize(len2 + 1);
    lastM.resize(len2 + 1);
    buffM.resize(len2 + 1);
    Ix.resize(len2 + 1);
    lastIx.resize(len2 + 1);
    Iy.resize(len2 + 1);
    lastIy.resize(len2 + 1);
    buffIy.resize(len2 + 1);
    int32_t score = processBlock(0, len2 - 1, 0, len1 - 1, 0, 0, 0, 0);
    makeAlignment(alignment);
    return score;
  }
};

} /* namespace PairAlign */


#endif /* MM_ALIGN_H */
