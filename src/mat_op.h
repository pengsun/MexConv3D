#pragma once
#include "mex.h"
#include "mex_shorthand.h"

// 2D matrix thin wrapper over raw data pointer
// presume continuous memory; 
// caller should assure the data are correctly assigned 
struct matw {
  float *beg;
  mwSize H, W;
};

// C += A * B or C = A * B (over write)
void CeqAxB (const matw &A, const matw &B, matw &C, bool isOverWrite = false);

// C += AT * B or C = AT * B (over write) 
void CeqATxB (const matw &A, const matw &B, matw &C, bool isOverWrite = false);

// C += A * BT or C = A * BT (over write)
void CeqAxBT (const matw &A, const matw &B, matw &C, bool isOverWrite = false);

