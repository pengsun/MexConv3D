#pragma once
#include "mex.h"

#ifdef VB
#define LOGMSG mexPrintf
#else
#define LOGMSG(...)
#endif // VB


inline size_t sizeofMxType (mxClassID t) {
  if (t == mxSINGLE_CLASS) return 4;
  if (t == mxINT32_CLASS)  return 4;
  if (t == mxDOUBLE_CLASS) return 8;

  return 0; // TODO: more types if necessary
}

inline size_t toMB (size_t n, mxClassID t) {
  return (n * sizeofMxType(t) / 1e6);
}

inline size_t toKB (size_t n, mxClassID t) {
  return (n * sizeofMxType(t) / 1e3);
}