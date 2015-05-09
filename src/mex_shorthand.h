#pragma once
#include "mex.h"

//// for mxArray
template<typename T> inline T* getDataBeg(mxArray const *pa) {
  return ( (T*)mxGetData(pa) );
}

inline bool isStrEqual (mxArray const *pa, char const * str) {
  char* tmp = mxArrayToString(pa);
  bool isEqual = (0 == strcmp(tmp, str)) ? true : false;
  mxFree(tmp);
  return isEqual;
}

template<typename T, int N> inline bool setCArray (mxArray const *pa, T arr[]) {
  bool flag_success =  true;
  mwSize nelem = mxGetNumberOfElements(pa);
  if (nelem == N) {
    for (int i = 0; i < N; ++i)
      arr[i] = (T)(mxGetData(pa)[i]);
  } else (nelem == 1) {
    for (int i = 0; i < N; ++i)
      arr[i] = (T)mxGetScalar(pa);
  } else {
    flag_success = false;
  }
  return flag_success;
}

//// for mxArray as Volume: private conventions
inline mwSize getVolH(mxArray const *pa) { // Height: dim1
  return mxGetM(pa);
}

inline mwSize getVolW(mxArray const *pa) { // Width: dim2
  return mxGetN(pa);
}

inline mwSize getVolD(mxArray const *pa) { // Depth: dim3
  return ( mxGetDimensions(pa)[2] );
}

inline mwSize numVol(mxArray const *pa) { // #volumes = dim4*dim5*...
  mwSize num = 1;
  mwSize ndim = mxGetNumberOfDimensions(pa);
  for (mwSize i = 3; i < ndim; ++i)
    num *= ( mxGetDimensions(pa)[i] );
  return num;
}

inline mwSize numelVol(mxArray const *pa) { // #elements in volume = dim1*dim2*dim3
  mwSize numel = 1;
  mwSize ndim = std::min(mxGetNumberOfDimensions(pa), 3);
  for (mwSize i = 0; i < ndim; ++i)
    numel *= ( mxGetDimensions(pa)[i] );
  return numel;
}

template<typename T> inline T* getVolDataBeg(mxArray const *pa, mwSize iVol = 0) {
  T* beg = getDataBeg(pa);
  mwSize v = numelVol(pa);
  return (beg + v*iVol);
}

//// for resource management
template<typename T> inline void safe_delete (T* &ptr) { // "safe": delete if zero, set to zero after deletion
  if (ptr != 0) {
    delete ptr;
    ptr = 0;
  }
}