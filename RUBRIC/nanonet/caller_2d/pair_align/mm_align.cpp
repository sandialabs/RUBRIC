#include <mm_align.h>

using namespace std;

namespace PairAlign {

int32_t MMAlign::processBlock(int xpos1, int xpos2, int ypos1, int ypos2,
                              int32_t m1, int32_t iy1, int32_t m2, int32_t iy2) {
  int len = xpos2 - xpos1 + 2;
  int mid = (ypos2 + ypos1 + 1) / 2;
  processUp(xpos1, xpos2, ypos1, mid, m1, iy1);
  lastM.swap(buffM);
  lastIy.swap(buffIy);
  processDown(xpos1, xpos2, mid, ypos2, m2, iy2);

  // Find alignment point.
  int pos = 0;
  int32_t maxScore = ZERO_PROB_SCORE;
  bool isMatch = false;
  for (int i = 0; i < len; ++i) {
    int dpos = len - i;
    int32_t mScore = ZERO_PROB_SCORE;
    if (i + xpos1 > 0) mScore = buffM[i] + lastM[dpos] - subMatrix(seq1[mid], seq2[i + xpos1 - 1]);
    int32_t deltay = openGapy;
    if (i + xpos1 == 0) deltay = startGapy;
    if (i + xpos1 == int(seq2.Size())) deltay = endGapy;
    int32_t yScore = buffIy[i] + lastIy[dpos - 1] - deltay;
    int32_t score = max(mScore, yScore);
    if (score > maxScore) {
      maxScore = score;
      pos = i;
      isMatch = (mScore >= yScore);
    }
  }

  // Push alignment position (if they aligned at this midline).
  if (isMatch) {
    matches.push_back(AlignPos(mid, xpos1 + pos - 1));
  }

  // Set up next blocks.
  int32_t newm1 = isMatch ? buffM[pos] : ZERO_PROB_SCORE;
  int32_t newiy1 = isMatch ? ZERO_PROB_SCORE : buffIy[pos];
  int dpos = len - pos;
  int32_t newm2 = isMatch ? lastM[dpos] : ZERO_PROB_SCORE;
  int32_t newiy2 = isMatch ? ZERO_PROB_SCORE : lastIy[dpos - 1];

  // Do new lower block.
  int newxpos2 = pos + xpos1 - 1;
  if (isMatch) --newxpos2;
  if (mid > ypos1 && newxpos2 >= xpos1) {
    processBlock(xpos1, newxpos2, ypos1, mid - 1, m1, iy1, newm2, newiy2);
  }

  // Do new upper block.
  int newxpos1 = pos + xpos1;
  if (mid < ypos2 && newxpos1 <= xpos2) {
    processBlock(pos + xpos1, xpos2, mid + 1, ypos2, newm1, newiy1, m2, iy2);
  }
  return maxScore;
}


void MMAlign::processUp(int xpos1, int xpos2, int ypos1, int ypos2,
                        int32_t m, int32_t iy) {
  View<int> view1 = View<int>(seq1, ypos1, ypos2 - ypos1 + 1, 1);
  View<int> view2 = View<int>(seq2, xpos1, xpos2 - xpos1 + 1, 1);
  int32_t gx1 = openGapx, gx2 = openGapx;
  int32_t hx1 = extendGapx, hx2 = extendGapx;
  if (ypos1 == 0) {
    gx1 = startGapx;
    hx1 = startGapx;
  }
  if (ypos2 == int(seq1.Size()) - 1) {
    gx2 = endGapx;
    hx2 = endGapx;
  }
  int32_t gy1 = openGapy, gy2 = openGapy;
  int32_t hy1 = extendGapy, hy2 = extendGapy;
  if (xpos1 == 0) {
    gy1 = startGapy;
    hy1 = startGapy;
  }
  if (xpos2 == int(seq2.Size()) - 1) {
    gy2 = endGapy;
    hy2 = endGapy;
  }
  int lenx = xpos2 - xpos1 + 2;
  int leny = ypos2 - ypos1 + 2;
  hmm(view1, view2, lenx, leny, m, iy, gx1, hx1, gx2, hx2, gy1, hy1, gy2, hy2);
}


void MMAlign::processDown(int xpos1, int xpos2, int ypos1, int ypos2,
                          int32_t m, int32_t iy) {
  View<int> view1 = View<int>(seq1, ypos2, ypos2 - ypos1 + 1, -1);
  View<int> view2 = View<int>(seq2, xpos2, xpos2 - xpos1 + 1, -1);
  int32_t gx1 = openGapx, gx2 = openGapx;
  int32_t hx1 = extendGapx, hx2 = extendGapx;
  if (ypos1 == 0) {
    gx2 = startGapx;
    hx2 = startGapx;
  }
  if (ypos2 == int(seq1.Size()) - 1) {
    gx1 = endGapx;
    hx1 = endGapx;
  }
  int32_t gy1 = openGapy, gy2 = openGapy;
  int32_t hy1 = extendGapy, hy2 = extendGapy;
  if (xpos1 == 0) {
    gy2 = startGapy;
    hy2 = startGapy;
  }
  if (xpos2 == int(seq2.Size()) - 1) {
    gy1 = endGapy;
    hy1 = endGapy;
  }
  int lenx = xpos2 - xpos1 + 2;
  int leny = ypos2 - ypos1 + 2;
  hmm(view1, view2, lenx, leny, m, iy, gx1, hx1, gx2, hx2, gy1, hy1, gy2, hy2);
}


void MMAlign::hmm(const View<int>& view1, const View<int>& view2, int lenx, int leny,
                  int32_t m, int32_t iy, int32_t gx1, int32_t hx1, int32_t gx2,
                  int32_t hx2, int32_t gy1, int32_t hy1, int32_t gy2, int32_t hy2) {
  lastM[0] = m;
  lastIy[0] = iy;
  lastIx[0] = ZERO_PROB_SCORE;
  for (int j = 1; j < lenx; ++j) {
    lastM[j] = ZERO_PROB_SCORE;
    lastIy[j] = ZERO_PROB_SCORE;
    if (j == 1) lastIx[j] = max(lastM[0], lastIy[0]) + gx1;
    else lastIx[j] = lastIx[j - 1] + hx1;
  }
  for (int i = 1; i < leny; ++i) {
    M[0] = ZERO_PROB_SCORE;
    Ix[0] = ZERO_PROB_SCORE;
    Iy[0] = max(lastIy[0] + hy1, lastM[0] + gy1);
    int32_t gx = (i == leny - 1) ? gx2 : openGapx;
    int32_t hx = (i == leny - 1) ? hx2 : extendGapx;
    for (int j = 1; j < lenx; ++j) {
      M[j] = TripleMax(lastM[j - 1], lastIx[j - 1], lastIy[j - 1]);
      M[j] += subMatrix(view1[i - 1], view2[j - 1]);
      int32_t gy = (j == lenx - 1) ? gy2 : openGapy;
      int32_t hy = (j == lenx - 1) ? hy2 : extendGapy;
      Iy[j] = TripleMax(lastM[j] + gy, lastIx[j] + gy, lastIy[j] + hy);
      Ix[j] = TripleMax(M[j - 1] + gx, Ix[j - 1] + hx, Iy[j - 1] + gx);
    }
    M.swap(lastM);
    Ix.swap(lastIx);
    Iy.swap(lastIy);
  }
}


void MMAlign::makeAlignment(vector<AlignPos>& alignment) {
  alignment.clear();
  sort(matches.begin(), matches.end());
  int lastx = -1, lasty = -1;
  for (size_t i = 0; i < matches.size(); ++i) {
    int x = matches[i].Pos2;
    int y = matches[i].Pos1;
    if (y > lasty + 1) {
      for (int p = lasty + 1; p < y; ++p) {
        alignment.push_back(AlignPos(p, -1));
      }
      lasty = y - 1;
    }
    if (x > lastx + 1) {
      for (int p = lastx + 1; p < x; ++p) {
        alignment.push_back(AlignPos(-1, p));
      }
      lastx = x - 1;
    }
    alignment.push_back(AlignPos(y, x));
    lastx = x;
    lasty = y;
  }
  if (lasty < int(seq1.Size()) - 1) {
    for (int p = lasty + 1; p < int(seq1.Size()); ++p) {
      alignment.push_back(AlignPos(p, -1));
    }
  }
  if (lastx < int(seq2.Size()) - 1) {
    for (int p = lastx + 1; p < int(seq2.Size()); ++p) {
      alignment.push_back(AlignPos(-1, p));
    }
  }
}


} /* namespace PairAlign */
