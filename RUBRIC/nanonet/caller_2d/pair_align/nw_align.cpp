#include <algorithm>
#include <nw_align.h>

using namespace std;


namespace PairAlign {

int32_t NWAlign::Align(const vector<int>& sequence1, const vector<int>& sequence2,
                                vector<AlignPos>& alignment) {
  int len1 = int(sequence1.size());
  int len2 = int(sequence2.size());
  diagScores.resize(len1 + 1, len2 + 1, false);
  upScores.resize(len1 + 1, len2 + 1, false);
  rightScores.resize(len1 + 1, len2 + 1, false);
  diagScores(0, 0) = 0;
  upScores(0, 0) = 0;
  rightScores(0, 0) = 0;
  // Fill in the left column. This is events from sequence 1 aligning before
  // the beginning of sequence 2.
  for (int i = 1; i <= len1; ++i) {
    upScores(i, 0) = upScores(i - 1, 0) + startGapy;
    diagScores(i, 0) = ZERO_PROB_SCORE;
    rightScores(i, 0) = ZERO_PROB_SCORE;
  }
  // Fill in the bottom row. This is events from sequence 2 aligning before
  // the beginning of sequence 1.
  for (int j = 1; j <= len2; ++j) {
    rightScores(0, j) = rightScores(0, j - 1) + startGapx;
    diagScores(0, j) = ZERO_PROB_SCORE;
    upScores(0, j) = ZERO_PROB_SCORE;
  }
  // Fill in the main body, but not the right column or top row.
  for (int i = 1; i < len1; ++i) {
    for (int j = 1; j < len2; ++j) {
      int32_t mismatch = subMatrix(sequence1[i - 1], sequence2[j - 1]);
      processNode(i, j, openGapx, extendGapx, openGapy, extendGapy, mismatch);
    }
  }
  // Fill in the top row. This is events from sequence 2 aligning to or after
  // the end of sequence 1.
  for (int j = 1; j < len2; ++j) {
    int32_t mismatch = subMatrix(sequence1[len1 - 1], sequence2[j - 1]);
    processNode(len1, j, endGapx, endGapx, openGapy, extendGapy, mismatch);
  }
  // Fill in the right column. This is events from sequence 1 aligning after
  // the end of sequence 2.
  for (int i = 1; i < len1; ++i) {
    int32_t mismatch = subMatrix(sequence1[i - 1], sequence2[len2 - 1]);
    processNode(i, len2, openGapx, extendGapx, endGapy, endGapy, mismatch);
  }
  // Fill in the top-right node.
  int32_t mismatch = subMatrix(sequence1[len1 - 1], sequence2[len2 - 1]);
  processNode(len1, len2, endGapx, endGapx, endGapy, endGapy, mismatch);
  backtrace(alignment);
  return TripleMax(diagScores(len1, len2), upScores(len1, len2), rightScores(len1, len2));
}


void NWAlign::backtrace(vector<AlignPos>& alignment) {
  alignment.clear();
  size_t i = diagScores.size1() - 1, j = diagScores.size2() - 1;
  while (i > 0 || j > 0) {
    int dir = TripleMaxIndex(diagScores(i, j), upScores(i, j), rightScores(i, j));
    switch(dir) {
    case 0:
      alignment.push_back(AlignPos(--i, --j));
      break;
    case 1:
      alignment.push_back(AlignPos(--i, -1));
      break;
    case 2:
      alignment.push_back(AlignPos(-1, --j));
      break;
    default:
      throw runtime_error("Error: Invalid result in backtrace.");
    }
  }
  reverse(alignment.begin(), alignment.end());
}


} /* namespace PairAlign */
