#include <stdexcept>
#include <numeric>
#include <stdint.h>
#include <bp_tools.h>
#include <view_numpy_arrays.h>
#include <utils.h>

namespace bp = boost::python;

bp::list check_vector(bp::list& v);
bp::list check_matrix(bp::list& m);
bp::dict check_map(bp::dict& d);
bp::numeric::array check_np_vector(bp::numeric::array& v, const std::string& t);
bp::numeric::array check_np_matrix(bp::numeric::array& m);
void check_exp(bp::numeric::array& data, bool fast);


/// Python class wrapper.
BOOST_PYTHON_MODULE(stub) {
  import_array();
  bp::numeric::array::set_module_and_type("numpy", "ndarray");
  def("check_vector", &check_vector);
  def("check_matrix", &check_matrix);
  def("check_map", &check_map);
  def("check_np_vector", &check_np_vector);
  def("check_np_matrix", &check_np_matrix);
  def("check_exp", &check_exp);
}


bp::list check_vector(bp::list& v) {
  std::vector<double> vec = list_to_vector<double>(v);
  bp::list out;
  for (size_t i = 0; i < vec.size(); ++i) {
    out.append(vec[i]);
  }
  return out;
}


bp::list check_matrix(bp::list& m) {
  ublas::matrix<double> mat = list_to_matrix<double>(m);
  bp::list out;
  for (size_t i = 0; i < mat.size1(); ++i) {
    bp::list row;
    for (size_t j = 0; j < mat.size2(); ++j) {
      row.append(mat(i, j));
    }
    out.append(row);
  }
  return out;
}


bp::dict check_map(bp::dict& d) {
  std::map<std::string, int> dm = dict_to_map<std::string, int>(d);
  bp::dict out;
  for (std::map<std::string, int>::iterator p = dm.begin(); p != dm.end(); ++p) {
    out[p->first] = p->second;
  }
  return out;
}


bp::numeric::array check_np_vector(bp::numeric::array& v, const std::string& t) {
  if (t == "int32") {
    VecView<int32_t> vec = view_1d_array<int32_t>(v);
    //std::cerr << "Stride for " << t << " is " << vec.stride() << std::endl;
    return vector_to_numpy(vec);
  }
  else if (t == "int64") {
    VecView<int64_t> vec = view_1d_array<int64_t>(v);
    //std::cerr << "Stride for " << t << " is " << vec.stride() << std::endl;
    return vector_to_numpy(vec);
  }
  else if (t == "float64") {
    VecView<double> vec = view_1d_array<double>(v);
    //std::cerr << "Stride for " << t << " is " << vec.stride() << std::endl;
    return vector_to_numpy(vec);
  }
  return new_numpy_1d<int>(0);
}


bp::numeric::array check_np_matrix(bp::numeric::array& m) {
  MatView<double> mat = view_2d_array<double>(m);
  return matrix_to_numpy(mat);
}


void check_exp(bp::numeric::array& data, bool fast) {
  VecView<float> x = view_1d_array<float>(data);
  size_t n = x.size();
  if (fast) {
    for (size_t i = 0; i < n; ++i) {
      x[i] = fastpow2(POW2FACTOR * x[i]);
    }
  }
  else {
    for (size_t i = 0; i < n; ++i) {
      x[i] = exp(x[i]);
    }
  }
}
