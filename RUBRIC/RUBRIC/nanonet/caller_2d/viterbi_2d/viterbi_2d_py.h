#ifndef VITERBI_2D_PY_H
#define VITERBI_2D_PY_H

#include <map>
#include <boost/python.hpp>
#include <boost/shared_ptr.hpp>
#include <bp_tools.h>
#include <view_numpy_arrays.h>
#include <viterbi_2d.h>


namespace bp = boost::python;


/// Viterbi 2D basecaller python wrapper.
class Viterbi2D_Py {
private:
  boost::shared_ptr<Viterbi2D> viterbi;
  boost::shared_ptr<Emission> emission1;
  boost::shared_ptr<Emission> emission2;
  std::vector<std::string> kmers;
  std::map<std::string, int> states;
  std::map<std::string, std::string> parms;
  int bandSize;
  bool useNoise;

  void setupKmers(int kmerLen);
  boost::shared_ptr<Emission> makeEmission(bp::dict& model, bool rc);
  boost::shared_ptr<Emission> dummyEmission(bp::list& kmers);
  void sortModel(std::vector<double>& levels, std::vector<double>& levelSpreads, std::vector<double>& noises,
                 std::vector<double>& noiseSpreads, std::vector<std::string>& mdlKmers, bool rc);
  void getEvents(bp::dict& events, std::vector<double>& means, std::vector<double>& stdvs,
                 std::vector<double>& stayWts, std::vector<double>& emWts);
  void makeBands(const Alignment& alignIn, std::vector<int32_t>& bandStarts, std::vector<int32_t>& bandEnds);
  bp::dict makeResult(const Alignment& alignOut, const std::vector<int16_t>& statesOut);

public:
  /** Constructor.
   *  @param[in] stateInfo Dictionary containing state information.
   *  @param[in] params Dictionary of basecalling parameters.
   *
   *  The state information should either contain 'model1' and 'model2'
   *  fields, containing the models for the template and complement data,
   *  or a 'kmers' field containing a list of the kmers (for posterior
   *  calling).
   */
  Viterbi2D_Py(bp::dict& stateInfo, bp::dict& params);

  /** Perform the basecall.
   *  @param[in] data1 Event sequence 1.
   *  @param[in] data2 Event sequence 2.
   *  @param[in] alignment Estimated alignment of sequence 1 to sequence 2.
   *  @param[in] prior The prior kmer for the "before alignment" node. None means no prior.
   *  @return Dictionary contain alignment and called kmers.
   */
  bp::dict Call(bp::dict& events1, bp::dict& events2, bp::list& alignment, bp::object& prior);

  /** Perform the basecall using posteriors.
   *  @param[in] post1 Posteriors for sequence 1.
   *  @param[in] post2 Posteriors for sequence 2.
   *  @param[in] stayWt1 Stay weights for sequence 1.
   *  @param[in] stayWt2 Stay weights for sequence 2.
   *  @param[in] alignment Estimated alignment of sequence 1 to sequence 2.
   *  @param[in] prior The prior kmer for the "before alignment" node. None means no prior.
   *  @return Dictionary contain alignment and called kmers.
   */
  bp::dict CallPost(bp::numeric::array& post1, bp::numeric::array& post2,
                    bp::numeric::array& stayWt1, bp::numeric::array& stayWt2,
                    bp::list& alignment, bp::object& prior);

  /// Get a list of the base transition probabilities.
  bp::list GetTransitionProbs() const;

  /// Get a list of kmers in operational order.
  bp::list GetKmerList() const;

  /// Get a list of the model levels for the first sequence.
  bp::numeric::array GetModelLevels1() const;

  /// Get a list of the model levels for the second sequence.
  bp::numeric::array GetModelLevels2() const;
};

/// Python class wrapper.
BOOST_PYTHON_MODULE(viterbi_2d) {
  import_array();
  bp::numeric::array::set_module_and_type("numpy", "ndarray");
  bp::class_<Viterbi2D_Py>("Viterbi2D", bp::init<bp::dict&, bp::dict&>())
    .def("call", &Viterbi2D_Py::Call)
    .def("call_post", &Viterbi2D_Py::CallPost)
    .def("get_kmer_list", &Viterbi2D_Py::GetKmerList)
    .def("get_model_levels1", &Viterbi2D_Py::GetModelLevels1)
    .def("get_model_levels2", &Viterbi2D_Py::GetModelLevels2);
  bp::scope().attr("ZERO_PROB_SCORE") = bp::object(ZERO_PROB_SCORE);
}


#endif /* VITERBI_2D_PY_H */
