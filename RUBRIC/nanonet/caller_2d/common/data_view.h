#ifndef DATA_VIEW_H
#define DATA_VIEW_H

#include <stdint.h>
#include <vector>
#include <iterator>
#include <boost/numeric/ublas/matrix.hpp>

namespace ublas = boost::numeric::ublas;


/** Represents a block of data as an STL-style container.
 *  This object does not own the data it views, and as such
 *  it can be invalidated if the data it views goes out of
 *  scope or is deleted. Note that the const-ness of this
 *  class protects the object itself, but not the data it
 *  views.
 **/
template <class T>
class VecView {
public:
  typedef T         value_type;
  typedef T*        pointer;
  typedef const T*  const_pointer;
  typedef size_t    size_type;
  typedef ptrdiff_t difference_type;
  typedef T&        reference;
  typedef const T&  const_reference;

protected:
  pointer ptr_;
  size_type size_;
  difference_type stride_;

public:
  class iterator : public std::iterator<std::random_access_iterator_tag, value_type, difference_type> {
  public:
      typedef T         value_type;
      typedef T*        pointer;
      typedef const T*  const_pointer;
      typedef ptrdiff_t difference_type;
      typedef T&        reference;
      typedef const T&  const_reference;

  protected:
    pointer ptr_;
    difference_type stride_;

  public:
    iterator() : ptr_(0), stride_(1) {}

    iterator(pointer p, difference_type s) : ptr_(p), stride_(s) {}

    operator void *() const {
      return ptr_;
    }

    reference operator[](int n) const {
      return *(ptr_ + n * stride_);
    }

    reference operator*() const {
      return *ptr_;
    }

    pointer operator->() const {
      return ptr_;
    }

    iterator& operator++() {
      ptr_ += stride_;
      return *this;
    }

    iterator& operator--() {
      ptr_ -= stride_;
      return *this;
    }

    iterator operator++(int) {
      iterator temp(*this);
      ptr_ += stride_;
      return temp;
    }

    iterator operator--(int) {
      iterator temp(*this);
      ptr_ -= stride_;
      return temp;
    }

    iterator& operator+=(int n) {
      ptr_ += n * stride_;
      return *this;
    }

    iterator& operator-=(int n) {
      ptr_ -= n * stride_;
      return *this;
    }

    iterator operator+(int n) const {
      return iterator(ptr_ + n * stride_, stride_);
    }

    iterator operator-(int n) const {
      return iterator(ptr_ - n * stride_, stride_);
    }

    difference_type operator-(const iterator& it) const {
      return (ptr_ - it.ptr_) / stride_;
    }

    bool operator==(const iterator& it) const {
      return (ptr_ == it.ptr_);
    }

    bool operator<(const iterator& it) const {
      return (stride_ > 0) ? (ptr_ < it.ptr_) : (it.ptr_ < ptr_);
    }

    bool operator>(const iterator& it) const {
      return (stride_ > 0) ? (ptr_ > it.ptr_) : (it.ptr_ > ptr_);
    }

    bool operator<=(const iterator& it) const {
      return (stride_ > 0) ? (ptr_ <= it.ptr_) : (it.ptr_ <= ptr_);
    }

    bool operator>=(const iterator& it) const {
      return (stride_ > 0) ? (ptr_ >= it.ptr_) : (it.ptr_ >= ptr_);
    }

    bool operator!=(const iterator& it) const {
      return (ptr_ != it.ptr_);
    }
  };

  typedef iterator const_iterator;

  /// Default constructor.
  VecView() : ptr_(0), size_(0), stride_(1) {}

  /** Basic view constructor.
   *  @param p Pointer to data to be viewed.
   *  @param len Number of elements to be viewed.
   *  @param stride Optional stride of elements.
   */
  VecView(const_pointer p, int len, int stride = 1) {
    view(p, len, stride);
  }

  /// View a std::vector.
  VecView(const std::vector<value_type>& vec) {
    view(vec);
  }

  /// Clears the current view.
  void clear() {
    ptr_ = 0;
    size_ = 0;
    stride_ = 1;
  }

  /** Basic view constructor.
   *  @param p Pointer to data to be viewed.
   *  @param len Number of elements to be viewed.
   *  @param stride Optional stride of elements.
   *
   *  The current view (if any) is abandoned.
   */
  void view(const_pointer p, int len, int stride = 1) {
    if (len < 0 || stride == 0) {
      throw std::runtime_error("Size must be >= 0, and stride cannot be zero.");
    }
    ptr_ = const_cast<pointer>(p);
    size_ = size_type(len);
    stride_ = difference_type(stride);
  }

  /** View a std::vector.
   *  The current view (if any) is abandoned.
   */
  void view(const std::vector<value_type>& vec) {
    if (vec.empty()) clear();
    else view(&vec[0], int(vec.size()), 1);
  }

  /** Return a new object that views a slice of the current one.
   *  @param start The starting position of the slice in the current view.
   *  @param len The number of elements to be viewed.
   *  @param stride Optional stride, which is relative to the current view.
   */
  VecView<T> slice(int start, int len, int stride = 1) const {
    if (start < 0) throw std::runtime_error("Slice cannot have negative start value.");
    if (stride > 0) {
      if (start + len * stride > int(size_)) throw std::runtime_error("Slice out of bounds.");
    }
    else if (stride < 0) {
      if (start >= int(size_) || start + (len - 1) * stride <= 0) throw std::runtime_error("Slice out of bounds.");
    }
    else {
      throw std::runtime_error("Slice cannot have zero stride.");
    }
    return VecView<T>(ptr_ + start * stride_, len, stride_ * stride);
  }

  /// The size of the data view.
  size_type size() const {
    return size_;
  }

  /// The stride of the view.
  difference_type stride() const {
    return stride_;
  }

  /// A pointer to the first element of the raw data of the view.
  pointer data() const {
    return ptr_;
  }

  /// Indexing operator.
  reference operator[](int n) const {
    return *(ptr_ + n * stride_);
  }

  /// Reverse the current view.
  void reverse() {
    if (ptr_ == 0 || size_ == 0) return;
    ptr_ += ptrdiff_t(size_ - 1) * stride_;
    stride_ = -stride_;
  }

  /// Iterator to start of view.
  iterator begin() const {
    return iterator(ptr_, stride_);
  }

  /// Iterator to past-the-end of the view.
  iterator end() const {
    return iterator(ptr_ + size_ * stride_, stride_);
  }
};


template <class T>
inline typename VecView<T>::iterator operator+(int n, const typename VecView<T>::iterator& it) {
  return it + n;
}


template <class T>
class MatView {
public:
  typedef T         value_type;
  typedef T*        pointer;
  typedef const T*  const_pointer;
  typedef size_t    size_type;
  typedef ptrdiff_t difference_type;
  typedef T&        reference;
  typedef const T&  const_reference;

  typedef typename VecView<value_type>::iterator iterator, const_iterator;

protected:
  pointer ptr_;
  size_type size1_, size2_;
  difference_type stride1_, stride2_;

public:

  MatView() : ptr_(0), size1_(0), size2_(0), stride1_(1), stride2_(1) {}

  MatView(const_pointer p, int len1, int len2, int stride1, int stride2) {
    view(p, len1, len2, stride1, stride2);
  }

  MatView(const_pointer p, int len1, int len2) {
    view(p, len1, len2);
  }

  MatView(const ublas::matrix<value_type>& mat) {
    view(mat);
  }

  void view(const_pointer p, int len1, int len2) {
    view(p, len1, len2, len2, 1);
  }

  void view(const_pointer p, int len1, int len2, int stride1, int stride2) {
    if (len1 < 0 || len2 < 0 || stride1 == 0 || stride2 == 0) {
      throw std::runtime_error("Lengths must be >= 0, and strides cannot be zero.");
    }
    ptr_ = const_cast<pointer>(p);
    size1_ = size_type(len1);
    size2_ = size_type(len2);
    stride1_ = difference_type(stride1);
    stride2_ = difference_type(stride2);
  }

  void view(const ublas::matrix<value_type>& mat) {
    int s1 = &mat(1, 0) - &mat(0, 0);
    int s2 = &mat(0, 1) - &mat(0, 0);
    view(&mat(0, 0), int(mat.size1()), int(mat.size2()), s1, s2);
  }

  reference operator()(int n, int m) const {
    return *(ptr_ + n * stride1_ + m * stride2_);
  }

  pointer data() const {
    return ptr_;
  }

  size_type size1() const {
    return size1_;
  }

  size_type size2() const {
    return size2_;
  }

  difference_type stride1() const {
    return stride1_;
  }

  difference_type stride2() const {
    return stride2_;
  }

  MatView<value_type> submatrix(int start1, int start2, int len1, int len2, int stride1 = 1, int stride2 = 1) const {
    return MatView<value_type>(ptr_ + start1 * stride1_ + start2 * stride2_, len1, len2, stride1 * stride1_, stride2 * stride2_);
  }

  VecView<value_type> row(int n) const {
    return VecView<value_type>(ptr_ + n * stride1_, size2_, stride2_);
  }

  VecView<value_type> column(int m) const {
    return VecView<value_type>(ptr_ + m * stride2_, size1_, stride1_);
  }

  void transpose() {
    if (ptr_ == 0 || size1_ == 0 || size2_ == 0) return;
    std::swap(stride1_, stride2_);
    std::swap(size1_, size2_);
  }

  iterator row_begin(int n) const {
    return iterator(ptr_ + n * stride1_, stride2_);
  }

  iterator row_end(int n) const {
    return iterator(ptr_ + n * stride1_ + size2_ * stride2_, stride2_);
  }

  iterator column_begin(int m) const {
    return iterator(ptr_ + m * stride2_, stride1_);
  }

  iterator column_end(int m) const {
    return iterator(ptr_ + m * stride2_ + size1_ * stride1_, stride1_);
  }
};


#endif /* DATA_VIEW_H */
