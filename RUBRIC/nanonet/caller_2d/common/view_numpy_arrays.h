#ifndef VIEW_NUMPY_ARRAYS_H
#define VIEW_NUMPY_ARRAYS_H

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <algorithm>
#include <stdint.h>
#include <boost/python.hpp>
#include <numpy/ndarrayobject.h>
#include <data_view.h>


namespace bp = boost::python;


inline int numpy_type(bool) {return NPY_BOOL;}
inline int numpy_type(int16_t) {return NPY_INT16;}
inline int numpy_type(int32_t) {return NPY_INT32;}
#ifdef _MSC_VER
  inline int numpy_type(int) {return NPY_INT32;}
#endif
inline int numpy_type(int64_t) {return NPY_INT64;}
inline int numpy_type(uint16_t) {return NPY_UINT16;}
inline int numpy_type(uint32_t) {return NPY_UINT32;}
inline int numpy_type(uint64_t) {return NPY_UINT64;}
inline int numpy_type(float) {return NPY_FLOAT32;}
inline int numpy_type(double) {return NPY_FLOAT64;}

template <class T>
inline int numpy_type(T&) {
  throw std::invalid_argument("Unknown type for numpy array.");
  return 0;
}


template <class T>
VecView<T> view_1d_array(const bp::numeric::array& arr) {
  PyArrayObject *obj = reinterpret_cast<PyArrayObject*>(arr.ptr());
  if (obj == 0) {
    throw std::invalid_argument("Could not covert bp::numeric::array to 1d numpy array.");
  }
  if (PyArray_DESCR(obj)->elsize != sizeof(T)) {
    throw std::invalid_argument("Numpy 1d array type does not match template type.");
  }
  if (PyArray_NDIM(obj) != 1) {
    throw std::length_error("Numpy array must be 1D.");
  }
  int length = PyArray_DIM(obj, 0);
  int stride = PyArray_STRIDE(obj, 0) / sizeof(T);
  npy_intp ind[1] = {0};
  T *data = reinterpret_cast<T*>(PyArray_GetPtr(obj, ind));
  return VecView<T>(data, length, stride);
}


template <class T>
MatView<T> view_2d_array(const bp::numeric::array& arr) {
  PyArrayObject *obj = reinterpret_cast<PyArrayObject*>(arr.ptr());
  if (obj == 0) {
    throw std::invalid_argument("Could not covert bp::numeric::array to 2d numpy array.");
  }
  if (PyArray_DESCR(obj)->elsize != sizeof(T)) {
    throw std::invalid_argument("Numpy 2d array type does not match template type.");
  }
  if (PyArray_NDIM(obj) != 2) {
    throw std::length_error("Numpy array must be 2D.");
  }
  int length1 = PyArray_DIM(obj, 0);
  int length2 = PyArray_DIM(obj, 1);
  int stride1 = PyArray_STRIDE(obj, 0) / sizeof(T);
  int stride2 = PyArray_STRIDE(obj, 1) / sizeof(T);
  npy_intp ind[2] = {0, 0};
  T *data = reinterpret_cast<T*>(PyArray_GetPtr(obj, ind));
  return MatView<T>(data, length1, length2, stride1, stride2);
}


template <class T>
bp::numeric::array new_numpy_1d(int n) {
  npy_intp dims[1] = {n};
  PyArrayObject *obj = reinterpret_cast<PyArrayObject*>(PyArray_SimpleNew(1, dims, numpy_type(T())));
  if (obj == 0) throw std::runtime_error("Call to PyArray_SimpleNew() failed.");
  bp::handle<> handle(reinterpret_cast<PyObject*>(obj));
  return bp::numeric::array(handle);
}


template <class T>
bp::numeric::array new_numpy_2d(int n, int m) {
  npy_intp dims[2] = {n, m};
  PyArrayObject *obj = reinterpret_cast<PyArrayObject*>(PyArray_SimpleNew(2, dims, numpy_type(T())));
  if (obj == 0) throw std::runtime_error("Call to PyArray_SimpleNew() failed.");
  bp::handle<> handle(reinterpret_cast<PyObject*>(obj));
  return bp::numeric::array(handle);
}


template <class T>
bp::numeric::array vector_to_numpy(const VecView<T>& vec) {
  bp::numeric::array arr = new_numpy_1d<T>(vec.size());
  VecView<T> lhs = view_1d_array<T>(arr);
  std::copy(vec.begin(), vec.end(), lhs.begin());
  return arr;
}


template <class T>
bp::numeric::array matrix_to_numpy(const MatView<T>& mat) {
  bp::numeric::array arr = new_numpy_2d<T>(mat.size1(), mat.size2());
  MatView<T> lhs = view_2d_array<T>(arr);
  for (size_t i = 0; i < mat.size1(); ++i) {
    std::copy(mat.row_begin(i), mat.row_end(i), lhs.row_begin(i));
  }
  return arr;
}


#endif /* VIEW_NUMPY_ARRAYS_H */
