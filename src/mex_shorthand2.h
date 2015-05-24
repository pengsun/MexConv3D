#pragma once

#include "mex.h"

#ifdef WITH_GPUARRAY
#include "gpu/mxGPUArray.h"
#else
struct mxGPUArray;
#endif // WITH_GPUARRAY

#include <cstring>
#include <numeric>
#include <cassert>


//// shorthand for mxArray
inline bool isStrEqual (mxArray const *pa, char const * str) {
  char* tmp = mxArrayToString(pa);
  bool isEqual = (0 == strcmp(tmp, str)) ? true : false;
  mxFree(tmp);
  return isEqual;
}

template<typename T> inline 
T* getDataBeg(mxArray const *pa) {
  return ( (T*)mxGetData(pa) );
}

template<typename T, int N> inline 
bool setCArray (mxArray const *pa, T arr[]) {
  if (!mxIsDouble(pa)) mexErrMsgTxt("setCArray: pa must be double matrix\n");

  bool flag_success =  true;
  mwSize nelem = mxGetNumberOfElements(pa);
  if (nelem == N) {
    for (int i = 0; i < N; ++i)
      arr[i] = T(*(getDataBeg<double>(pa) + i));
  } else if (nelem == 1) {
    for (int i = 0; i < N; ++i)
      arr[i] = T(mxGetScalar(pa));
  } else {
    flag_success = false;
  }
  return flag_success;
}


//// Thin wrapper for mxArray, which can be mxGPUArray
struct xpuMxArrayTW {
  enum DEV_TYPE {CPU, GPU};

  xpuMxArrayTW  ();
  ~xpuMxArrayTW ();

  void     setMxArray (mxArray *pa); // never owns the data
  mxArray* getMxArray ();

  mwSize    getNDims     ();
  mwSize    getSizeAtDim (mwSize dim);
  DEV_TYPE  getDevice    ();
  mxClassID getElemType  ();
  void*     getDataBeg   ();

// private:
  mxArray*    pa_cpu;
  mxGPUArray* pa_gpu;
  DEV_TYPE    dt;
};

//// Shorthand for xpuMxArrayTW
mxArray* createVol5d (mwSize sz[], xpuMxArrayTW::DEV_TYPE dt);

mxArray* createVol5dZeros (mwSize sz[], xpuMxArrayTW::DEV_TYPE dt);

mxArray* createVol5dLike (const xpuMxArrayTW &rhs, mxClassID tp = mxSINGLE_CLASS);

template<typename T> inline 
T* getVolDataBeg(const xpuMxArrayTW &rhs, mwSize iVol = 0) {
  return 0;
}

template<typename T> inline 
T* getVolInstDataBeg(const xpuMxArrayTW &rhs, mwSize iInst = 0) {
  return 0;
}

inline mwSize numVol (const xpuMxArrayTW &rhs) {
  return 0;
}

inline mwSize numel (const xpuMxArrayTW &rhs) {
  return 0;
}


//// Miscellaneous
template<typename T> inline 
void safe_delete (T* &ptr) { // "safe": delete if zero, set to zero after deletion
  if (ptr != 0) {
    delete ptr;
    ptr = 0;
  }
}
