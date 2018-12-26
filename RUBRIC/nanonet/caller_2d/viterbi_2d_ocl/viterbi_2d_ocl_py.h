#ifndef VITERBI_2D_OCL_PY_H
#define VITERBI_2D_OCL_PY_H

#include <map>
#include <string>
#include <vector>
#include <boost/python.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include <bp_tools.h>
#include <view_numpy_arrays.h>

#include "proxyCL.h"
#include "viterbi_2d_ocl.h"


namespace bp = boost::python;

/// proxyCL python wrapper.
class proxyCL_Py : public proxyCL
{
public:
  bp::tuple available_vendors() const
  {
    std::string error;
    std::vector <vendor> vendors = proxyCL::available_vendors(error);
    return bp::make_tuple(vendors, error);
  }

  bp::tuple available_vendors_str() const
  {
    std::string error;
    std::vector <std::string> vendors = proxyCL::available_vendors_str(error);
    return bp::make_tuple(vendors, error);
  }

  bp::tuple available_vendors_str_ex() const
  {
    std::string error;
    std::vector <std::string> vendors = proxyCL::available_vendors_str_ex(error);
    return bp::make_tuple(vendors, error);
  }

  bp::tuple select_vendor(vendor v)
  {
    std::string error;
    bool ret = proxyCL::select_vendor(v, error);
    return bp::make_tuple(ret, error);
  }

  bp::tuple select_vendor_str(const std::string &vendor)
  {
    std::string error;
    bool ret = proxyCL::select_vendor(vendor, error);
    return bp::make_tuple(ret, error);
  }

  bp::tuple create_context(device_type type = undefined)
  {
    bool ret;
    std::string error;
    if (type == undefined)
    {
        ret = proxyCL::create_context(error);
    }
    else
    {
        ret = proxyCL::create_context(type, error);
    }
    return bp::make_tuple(ret, error);
  }

  bp::tuple available_devices() const
  {
    std::string error;
    std::vector <device_info> devices = proxyCL::available_devices(error);
    return bp::make_tuple(devices, error);
  }

  bp::tuple select_device(size_t id)
  {
    std::string error;
    bool ret = proxyCL::select_device(id, error);
    return bp::make_tuple(ret, error);
  }

  bp::tuple get_device_info(size_t id) const
  {
    std::string error;
    std::string info = proxyCL::get_device_info(id, error);
    return bp::make_tuple(info, error);
  }
};

BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(create_context_overloads, create_context, 0, 1);

/// Viterbi 2D basecaller python wrapper.
class Viterbi2Docl_Py {
private:
  boost::shared_ptr<Viterbi2Docl> viterbi;
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
   *  @param[in] proxy_cl Initialised proxyCL object.
   */
  Viterbi2Docl_Py(proxyCL_Py& proxy_cl);

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

  /** Initialize OpenCL kernel and command queue. This can also be used to just build the binary kernel file.
   *  @param[in] model1 Model to use for first sequence of events.
   *  @param[in] model2 Model to use for second sequence of events.
   *  @param[in] params Dictionary of basecalling parameters.
   */
  bp::tuple InitCL(const std::string& srcKernelDir, const std::string& binKernelDir,
    bool enable_fp64, size_t num_states, size_t work_group_size);

  /** Initialize model data and basecalling parameters. This is not necessary when just creating the binary kernel file.
   *  @param[in] stateInfo Dictionary containing state information.
   *  @param[in] params Dictionary of basecalling parameters.
   *
   *  The state information should either contain 'model1' and 'model2'
   *  fields, containing the models for the template and complement data,
   *  or a 'kmers' field containing a list of the kmers (for posterior
   *  calling).
   */
  void InitData(bp::dict& stateInfo, bp::dict& params);

  /// Get a list of kmers in operational order.
  bp::list GetKmerList() const;

  /// Get a list of the model levels for the first sequence.
  bp::numeric::array GetModelLevels1() const;

  /// Get a list of the model levels for the second sequence.
  bp::numeric::array GetModelLevels2() const;
};

/// Python class wrapper.
BOOST_PYTHON_MODULE(viterbi_2d_ocl) {
  import_array();
  bp::numeric::array::set_module_and_type("numpy", "ndarray");

  bp::scope().attr("ZERO_PROB_SCORE") = bp::object(ZERO_PROB_SCORE);

  bp::enum_<vendor>("vendor")
    .value("amd",       vendor::amd)
    .value("intel",     vendor::intel)
    .value("nvidia",    vendor::nvidia)
    .value("apple",     vendor::apple)
    .value("other",     vendor::other)
    ;

  bp::enum_<device_type>("device_type")
    .value("cpu", device_type::cpu)
    .value("gpu", device_type::gpu)
    .value("all", device_type::all)
    ;

  bp::class_<device_info>("device_info", bp::no_init)
    .def_readonly("id",     &device_info::id)
    .def_readonly("name",   &device_info::name)
    .def_readonly("type",   &device_info::type)
    ;

  bp::class_<std::vector<vendor> >("vendor_vec")
    .def(bp::vector_indexing_suite<std::vector<vendor> >())
    ;

    bp::class_<std::vector<device_info> >("device_info_vec")
    .def(bp::vector_indexing_suite<std::vector<device_info> >())
    ;

  bp::class_<proxyCL_Py, boost::noncopyable>("proxyCL", bp::init<>())
    .def("available_vendors",       &proxyCL_Py::available_vendors)
    .def("available_vendors_str",   &proxyCL_Py::available_vendors_str)
    .def("available_vendors_str_ex",&proxyCL_Py::available_vendors_str_ex)
    .def("enable_cuda_build_cache", &proxyCL_Py::enable_cuda_build_cache)
    .def("select_vendor",           &proxyCL_Py::select_vendor)
    .def("select_vendor_str",       &proxyCL_Py::select_vendor_str)
    .def("create_context",          &proxyCL_Py::create_context, create_context_overloads())
    .def("available_devices",       &proxyCL_Py::available_devices)
    .def("get_device_info",         &proxyCL_Py::get_device_info)
    .def("select_device",           &proxyCL_Py::select_device)
    ;

  bp::class_<Viterbi2Docl_Py>("Viterbi2Docl", bp::init<proxyCL_Py&>())
    .def("call",                    &Viterbi2Docl_Py::Call)
    .def("call_post",               &Viterbi2Docl_Py::CallPost)
    .def("init_cl",                 &Viterbi2Docl_Py::InitCL)
    .def("init_data",               &Viterbi2Docl_Py::InitData)
    .def("get_kmer_list",           &Viterbi2Docl_Py::GetKmerList)
    .def("get_model_levels1",       &Viterbi2Docl_Py::GetModelLevels1)
    .def("get_model_levels2",       &Viterbi2Docl_Py::GetModelLevels2)
    ;
}


#endif /* VITERBI_2D_OCL_PY_H */
