#pragma once
#include "tmwtypes.h"

inline size_t ceil_divide (size_t a, size_t b) {
  return (a + b - 1)/b;
}

const int CU_NUM_THREADS = 2048; 