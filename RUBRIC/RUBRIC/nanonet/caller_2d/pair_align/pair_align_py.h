#ifndef PAIR_ALIGN_PY_H
#define PAIR_ALIGN_PY_H

#include <boost/python.hpp>
#include <boost/shared_ptr.hpp>
#include <pair_align.h>


namespace bp = boost::python;
namespace ublas = boost::numeric::ublas;


/// Boost-Python wrapper class for pairwise alignment classes.
class PairAlign_Py {
private:
  ublas::matrix<int32_t> subMatrix;
  boost::shared_ptr<PairAlign::Aligner> aligner;

public:
  /** Constructor
   *  @param[in] subMat Substitution matrix. List of lists.
   *  @param[in] gapPenalties List of gap penalties (length 8).
   *  @param[in] lowmem Flag indicating whether to use the faster Neddleman-Wunsch
   *             implementation or the slower linear-memory Myers-Miller
   *             implementation.
   *
   *  The 8 values in the gap penalty list should be as follows:
   *    start_gap1  Penalty for aligning to a gap before sequence 1.
   *    end_gap1    Penalty for aligning to a gap after sequence 1.
   *    open_gap1   Penalty for opening a gap within sequence 1.
   *    extend_gap1 Penalty for extending a gap within sequence 1.
   *    start_gap2  Penalty for aligning to a gap before sequence 2.
   *    end_gap2    Penalty for aligning to a gap after sequence 2.
   *    open_gap2   Penalty for opening a gap within sequence 2.
   *    extend_gap2 Penalty for extending a gap within sequence 2.
   */
  PairAlign_Py(bp::list& subMat, bp::list& gapPenalties, bool lowmem);

  /** Align two sequences.
   *  @param[in] sequence1 First sequence of states.
   *  @param[in] sequence2 Second sequence of states.
   *  @return Tuple containing a list holding the resulting alignment
   *      and an alignment score. This is not normalized.
   *
   *  The alignment object will contain one entry per alignment position. Each entry
   *  is a tuple containing the indexes of the two sequence elements that align to that
   *  position. If one sequence has aligned to a gap at that position, the value for
   *  the other sequence will be -1.
   *
   *  Note that the two sequences must contain only values from 0 to n-1, where n is the
   *  size of the nxn substitution matrix.
   */
  bp::tuple Align(bp::list& sequence1, bp::list& sequence2);
};



/// Python class wrapper.
BOOST_PYTHON_MODULE(pair_align) {
  bp::class_<PairAlign_Py>("Aligner", bp::init<bp::list&, bp::list&, bool>(bp::args("sub_matrix", "gap_penalties", "lowmem")))
    .def("align", &PairAlign_Py::Align);
}


#endif /* PAIR_ALIGN_PY_H */
