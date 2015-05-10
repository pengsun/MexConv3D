#pragma once
#include "mex.h"
#include <cstring>
#include <numeric>
#include <cassert>


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
  if (!mxIsDouble(pa)) mexErrMsgTxt("setCArray: pa must be double mtrix\n");

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


//// for mxArray: number of elements, dimensions, etc
template<mwSize DIM>
inline mwSize getSzAtDim(mxArray const *pa) { // DIM is matlab 1-base
  if (mxGetNumberOfDimensions(pa) < DIM)
    return (mwSize)1;
  else
    return ( mxGetDimensions(pa)[DIM-1] );
}

//// for mxArray as Volume: private conventions
inline mwSize getVolH(mxArray const *pa) { // Height: dim1
  return mxGetM(pa);
}

inline mwSize getVolW(mxArray const *pa) { // Width: dim2
  return getSzAtDim<2>(pa);
}

inline mwSize getVolD(mxArray const *pa) { // Depth: dim3
  return getSzAtDim<3>(pa);
}

inline mwSize getVolP(mxArray const *pa) { // P: dim4
  return getSzAtDim<4>(pa);
}

inline mwSize getVolM(mxArray const *pa) { // M: dim4 (alias of P)
  return getSzAtDim<4>(pa);
}

inline mwSize getVolQ(mxArray const *pa) { // Q: dim5
  return getSzAtDim<5>(pa);
}

inline mwSize getVolN(mxArray const *pa) { // N: dim5 (alias of Q)
  return getSzAtDim<5>(pa);
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
  mwSize ndim = std::min(mxGetNumberOfDimensions(pa), (mwSize)3);
  for (mwSize i = 0; i < ndim; ++i)
    numel *= ( mxGetDimensions(pa)[i] );
  return numel;
}

template<typename T> inline T* getVolDataBeg(mxArray const *pa, mwSize iVol = 0) {
  T* beg = getDataBeg<T>(pa);
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

inline mxArray* createVol5d (mwSize sz[], mxClassID tp = mxSINGLE_CLASS) {
  mwSize ndim = 5;
  return mxCreateNumericArray(ndim, sz, tp, mxREAL);
}

inline mxArray* createVol5d (mwSize H, mwSize W, mwSize D, mwSize P, mwSize Q, 
                             mxClassID tp = mxSINGLE_CLASS) 
{
  mwSize sz[5];
  sz[0] = H; sz[1] = W; sz[2] = D; sz[3] = P; sz[4] = Q;
  return createVol5d(sz, tp);
}

inline mxArray* createVol5dZeros (mwSize sz[], mxClassID tp = mxSINGLE_CLASS) { // create and ensure zeros
  return createVol5d (sz, tp); // matlab mxCreate* ensures zeros, see matlab doc 
}

inline mxArray* createVol5dZeros (mwSize H, mwSize W, mwSize D, mwSize P, mwSize Q, 
                                  mxClassID tp = mxSINGLE_CLASS) 
{
  return createVol5d(H,W,D,P,Q, tp); // matlab mxCreate* ensures zeros, see matlab doc 
}

inline mxArray* createVol5dLike (mxArray const *pa, mxClassID tp = mxUNKNOWN_CLASS) { // take pa's type if unset
  if (tp == mxUNKNOWN_CLASS)
    return mxCreateNumericArray(mxGetNumberOfDimensions(pa), 
                                mxGetDimensions(pa), mxGetClassID(pa), mxREAL);
  else 
    return mxCreateNumericArray(mxGetNumberOfDimensions(pa), 
                                mxGetDimensions(pa), tp, mxREAL);
}