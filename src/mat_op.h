#pragma once
#include "mex.h"
#include "mex_shorthand.h"

// 2D matrix thin wrapper over raw data pointer
// presume continuous memory
// caller owns data and assures assignment 
struct matw {
  float *beg;
  mwSize H, W;
};

// A*B + C -> C (accumulation) or A*B -> C(overwrite)
void AxBtoC (const matw &A, const matw &B, matw &C, bool isOverWrite);

// AT*B + C -> C (accumulation) or AT*B -> C (overwrite) 
void ATxBtoC (const matw &A, const matw &B, matw &C, bool isOverWrite);

// A*BT + C -> C (accumulation) or A*BT -> C (overwrite)
void AxBTtoC (const matw &A, const matw &B, matw &C, bool isOverWrite);

