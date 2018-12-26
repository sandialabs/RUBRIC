#include <Python.h>

#include <cassert>
#include <cstdlib>
#include <limits>
#include <vector>

#define MODULE_API_EXPORTS
#include "module.h"
#include "stdint.h"

#include <iostream>

typedef float ftype;
using namespace std;


static PyMethodDef DecodeMethods[] = {
    {NULL, NULL, 0, NULL}        /* Sentinel */
};

PyMODINIT_FUNC initnanonetdecode(void) {
    (void) Py_InitModule("nanonetdecode", DecodeMethods);
}


extern "C" void viterbi_update(
  ftype* vit_last, ftype* vit_curr, int32_t* max_idx,
  const size_t num_bases, const size_t num_kmers,
  const ftype stay, const ftype step, const ftype skip, const ftype slip
){

  for ( size_t kmer=0 ; kmer<num_kmers ; kmer++){
    max_idx[kmer] = -1;
    vit_curr[kmer] = -std::numeric_limits<ftype>::infinity();
  }

  // Stay
  for ( size_t kmer=0 ; kmer<num_kmers ; kmer++){
    if(vit_last[kmer]+stay>vit_curr[kmer]){
      vit_curr[kmer] = vit_last[kmer]+stay;
      max_idx[kmer] = kmer;
    }
  }
  // Step
  for ( size_t kmer=0 ; kmer<num_kmers ; kmer++){
    const size_t idx = (kmer*num_bases)%num_kmers;
    for ( size_t i=0 ; i<num_bases ; i++){
      if(vit_last[kmer]+step>vit_curr[idx+i]){
        vit_curr[idx+i] = vit_last[kmer]+step;
        max_idx[idx+i] = kmer;
      }
    }
  }
  // Skip
  for ( size_t kmer=0 ; kmer<num_kmers ; kmer++){
    const size_t idx = (kmer*num_bases*num_bases)%num_kmers;
    for ( size_t i=0 ; i<num_bases*num_bases ; i++){
      if(vit_last[kmer]+skip>vit_curr[idx+i]){
        vit_curr[idx+i] = vit_last[kmer]+skip;
        max_idx[idx+i] = kmer;
      }
    }
  }
  // Slip
  if (slip > -std::numeric_limits<ftype>::infinity()){
    ftype slip_max = -std::numeric_limits<ftype>::infinity();
    size_t slip_idx = 0;
    for ( size_t kmer=0 ; kmer<num_kmers ; kmer++){
      if(vit_last[kmer]+slip>slip_max){
        slip_max = vit_last[kmer]+slip;
        slip_idx = kmer;
      }
    }
    for ( size_t kmer=0 ; kmer<num_kmers ; kmer++){
      if(slip_max>vit_curr[kmer]){
        vit_curr[kmer] = slip_max;
        max_idx[kmer] = slip_idx;
      }
    }
  }
}


extern "C" MODULE_API ftype decode_path(ftype * logpost, const size_t num_events, const size_t num_bases, const size_t num_kmers){
  assert(NULL!=logpost);
  assert(num_events>0);
  assert(num_bases>0);
  assert(num_kmers>0);

  std::vector<int32_t> max_idx(num_kmers);
  std::vector<ftype> vit_last(num_kmers);
  std::vector<ftype> vit_curr(num_kmers);

  // Treat all movement types equally, disallow slip (allowing slip
  //   would simply give kmer with maximum posterioir)
  ftype stay = 0.0;
  ftype step = 0.0;
  ftype skip = 0.0;
  ftype slip = -std::numeric_limits<ftype>::infinity();

  // Initial values
  for ( size_t kmer=0 ; kmer<num_kmers ; kmer++){
    vit_last[kmer] = logpost[kmer];
  }

  for ( size_t ev=1 ; ev<num_events ; ev++){
    const size_t idx1 = (ev-1)*num_kmers;
    const size_t idx2 = ev*num_kmers;

    viterbi_update(
      &vit_last[0], &vit_curr[0], &max_idx[0], //.data() not supported on VSC++ for python,
      num_bases, num_kmers,
      stay, step, skip, slip
    );

    // Emission
    for ( size_t kmer=0 ; kmer<num_kmers ; kmer++){
      vit_curr[kmer] += logpost[idx2+kmer];
    }

    // Traceback information
    for ( size_t kmer=0 ; kmer<num_kmers ; kmer++){
      logpost[idx1+kmer] = max_idx[kmer];
    }
    std::swap( vit_last, vit_curr );
  }

  // Decode states
  // Last state by Viterbi matrix
  const size_t idx = (num_events-1)*num_kmers;
  ftype max_val = -std::numeric_limits<ftype>::infinity();
  int max_kmer = -1;
  for ( size_t kmer=0 ; kmer<num_kmers ; kmer++){
    if(vit_last[kmer]>max_val){
      max_val = vit_last[kmer];
      max_kmer = kmer;
    }
  }
  logpost[idx] = max_kmer;
  // Other states by traceback
  for ( size_t ev=(num_events-1) ; ev>0 ; ev--){
    const size_t idx = (ev-1)*num_kmers;
    logpost[idx] = logpost[idx+(int)logpost[idx+num_kmers]];
  }

  return max_val;
}


extern "C" MODULE_API void estimate_transitions(ftype* post, ftype* trans, const size_t num_events, const size_t num_bases, const size_t num_kmers){
  assert(NULL!=post);
  assert(num_events>0);
  assert(num_bases>0);
  assert(num_kmers>0);
  const size_t num_bases_sq = num_bases * num_bases;

  for (size_t ev = 1; ev < num_events; ++ev) {
    ftype stay_sum = 0.f;
    ftype step_sum = 0.f;
    ftype skip_sum = 0.f;
    const size_t idx1 = ev * num_kmers;
    const size_t idx0 = idx1 - num_kmers; 
    for (size_t i = 0; i < num_kmers / num_bases_sq; ++i) {
      ftype sum16 = 0.f;
      for (size_t j = 0; j < num_bases; ++j) {
        ftype sum4 = 0.f;
        for (size_t k = 0; k < num_bases; ++k) {
          size_t kmer = i * num_bases_sq + j * num_bases + k;
          ftype p = post[idx1 + kmer];
          stay_sum += post[idx0 + kmer] * p;
          sum4 += p;
        }
        for (size_t step_from = num_bases * i + j; step_from < num_kmers; step_from += num_kmers / num_bases) {
          step_sum += sum4 * post[idx0 + step_from];
        }
        sum16 += sum4;
      }
      for (size_t skip_from = i; skip_from < num_kmers; skip_from += num_kmers / num_bases_sq) {
        skip_sum += sum16 * post[idx0 + skip_from];
      }
    }
    step_sum *= 0.25f;
    skip_sum *= 0.0625f;
    trans[(ev-1) * 3] = stay_sum;
    trans[(ev-1) * 3 + 1] = step_sum;
    trans[(ev-1) * 3 + 2] = skip_sum;
  }
}
