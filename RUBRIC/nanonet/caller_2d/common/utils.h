#ifndef UTILS_H
#define UTILS_H

#include <emmintrin.h>
#include <stdint.h>
#include <iostream>
#include <cmath>
#include <vector>
#include <numeric>


/// 1.0 / ln(2) - Needed as scaling factor for computing exp(x) from 2^x.
static const float POW2FACTOR = 1.442695040f;

inline float square(float x) {return x * x;}
inline float cube(float x) {return x * x * x;}


/// Fast approximation for computing 2^p in single precision.
inline float fastpow2(float p) {
  float clipp = (p > -125.0f) ? p : -125.0f;
  union {uint32_t i; float f;} v = {uint32_t((1 << 23) * (clipp + 126.94269504f))};
  return v.f;
}

/// Fast vectorized approximation for computing 2^p in single precision for 4 numbers.
inline __m128 vfasterpow2(const __m128 p) {
    const __m128 c_126_94269504 = _mm_set_ps1(126.94269504f);
    const __m128 lt125 = _mm_cmplt_ps(p, _mm_set_ps1(-125.0f));
    const __m128 clipp = _mm_or_ps(_mm_andnot_ps(lt125, p), _mm_and_ps(lt125, _mm_set_ps1(-125.0f)));
    union { __m128i i; __m128 f; } v = { _mm_cvttps_epi32(_mm_mul_ps(_mm_set_ps1(1 << 23), _mm_add_ps(clipp, c_126_94269504))) };
    return v.f;
}

/** Generic normalization function.
 *  @param NUM_STATES The number of states to normalize over.
 *  @param data An array of floats.
 *  @returns The normalization factor used.
 */
template <int NUM_STATES>
float normalize(float *data) {
  float sum = 0.0f;
  for (int state = 0; state < NUM_STATES; ++state) {
    sum += data[state];
  }
  if (sum < 1e-38f || !std::isfinite(sum)) {
    throw std::runtime_error("Normalization error.");
  }
  float norm = 1.0f / (sum);
  for (int state = 0; state < NUM_STATES; ++state) {
    data[state] *= norm;
  }
  return sum;
}


#endif /* UTILS_H */
