#pragma once
#include "tmwtypes.h"

inline size_t ceil_divide (size_t a, size_t b) {
  return (a + b - 1)/b;
}


#if __CUDA_ARCH__ >= 200
const int CU_NUM_THREADS = 1024; 
#else
const int CU_NUM_THREADS = 512; 
#endif