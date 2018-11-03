#ifndef BP_TOOLS_H
#define BP_TOOLS_H

#include <map>
#include <vector>
#include <boost/python.hpp>
#include <boost/numeric/ublas/matrix.hpp>

namespace ublas = boost::numeric::ublas;
namespace bp = boost::python;


/// Construct a std::vector from a python list. Elements must match template type.
template <class T>
std::vector<T> list_to_vector(const bp::list& in) {
  int count = bp::len(in);
  std::vector<T> out(count);
  for (int i = 0; i < count; ++i) {
    out[i] = bp::extract<T>(in[i]);
  }
  return out;
}


/// Construct a std::vector of std::pair objects from a python list of tuples.
template <class T>
std::vector<std::pair<T, T> > list_to_pair_vector(const bp::list& in) {
  int count = bp::len(in);
  std::vector<std::pair<T, T> > out(count);
  for (int i = 0; i < count; ++i) {
    bp::tuple p = bp::extract<bp::tuple>(in[i]);
    T first = bp::extract<T>(p[0]);
    T second = bp::extract<T>(p[1]);
    out[i] = std::make_pair(first, second);
  }
  return out;
}


/// Construct an ublas::matrix from a python list of lists. Elements must match template type.
template <class T>
ublas::matrix<T> list_to_matrix(const bp::list& in) {
  int nrows = bp::len(in);
  int ncols = bp::len(bp::extract<bp::list>(in[0]));
  ublas::matrix<T> out(nrows, ncols);
  for (int i = 0; i < nrows; ++i) {
    bp::list row = bp::extract<bp::list>(in[i]);
    if (bp::len(row) != ncols) {
      throw std::runtime_error("Error: Not all columns are the same length.");
    }
    for (int j = 0; j < ncols; ++j) {
      out(i, j) = bp::extract<T>(row[j]);
    }
  }
  return out;
}


/// Construct a std::map from a python dictionary.
template <class KEY, class VAL>
std::map<KEY, VAL> dict_to_map(const bp::dict& in) {
  bp::list items = in.items();
  int count = bp::len(items);
  std::map<KEY, VAL> out;
  for (int i = 0; i < count; ++i) {
    bp::tuple pair = bp::extract<bp::tuple>(items[i]);
    KEY key = bp::extract<KEY>(pair[0]);
    VAL val = bp::extract<VAL>(pair[1]);
    out[key] = val;
  }
  return out;
}


#endif /* BP_TOOLS_H */
