#include <pair_align_py.h>
#include <nw_align.h>
#include <mm_align.h>

using namespace std;
using ublas::matrix;


template <class T>
void list_to_vector(bp::list& in, vector<T>& out) {
  out.clear();
  int count = bp::len(in);
  out.resize(count);
  for (int i = 0; i < count; ++i) {
    out[i] = bp::extract<T>(in[i]);
  }
}

template <class T>
void list_to_matrix(bp::list& in, matrix<T>& out) {
  out.clear();
  int nrows = bp::len(in);
  int ncols = bp::len(bp::extract<bp::list>(in[0]));
  out.resize(nrows, ncols);
  for (int i = 0; i < nrows; ++i) {
    bp::list row = bp::extract<bp::list>(in[i]);
    if (bp::len(row) != ncols) {
      throw runtime_error("Error: Not all columns are the same length.");
    }
    for (int j = 0; j < ncols; ++j) {
      out(i, j) = bp::extract<T>(row[j]);
    }
  }
}


PairAlign_Py::PairAlign_Py(bp::list& subMat, bp::list& gapPen, bool lowmem) {
  list_to_matrix(subMat, subMatrix);
  vector<int> gapPenalties;
  list_to_vector(gapPen, gapPenalties);
  if (lowmem) {
    aligner = boost::shared_ptr<PairAlign::Aligner>(new PairAlign::MMAlign(subMatrix, gapPenalties));
  }
  else {
    aligner = boost::shared_ptr<PairAlign::Aligner>(new PairAlign::NWAlign(subMatrix, gapPenalties));
  }
}


bp::tuple PairAlign_Py::Align(bp::list& sequence1, bp::list& sequence2) {
  vector<PairAlign::AlignPos> alignVec;
  vector<int> seq1, seq2;
  list_to_vector(sequence1, seq1);
  list_to_vector(sequence2, seq2);
  int32_t score = aligner->Align(seq1, seq2, alignVec);
  bp::list alignment;
  for (size_t i = 0; i < alignVec.size(); ++i) {
    alignment.append(bp::make_tuple(alignVec[i].Pos1, alignVec[i].Pos2));
  }
  return bp::make_tuple(alignment, score);
}
