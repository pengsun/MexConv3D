#include "mex_shorthand2.h"


//// Impl of xpuMxArray
xpuMxArrayTW::xpuMxArrayTW()
{
  pa_cpu = 0;
  pa_gpu = 0;
  dt = CPU;
}

xpuMxArrayTW::~xpuMxArrayTW()
{
  pa_cpu = 0;
#ifdef WITH_GPUARRAY
  if (dt == GPU) {// always do this according to Matlab Doc
    mxGPUDestroyGPUArray(pa_gpu);
    pa_gpu = 0;
  }
#endif // WITH_GPUARRAY
}

mwSize xpuMxArrayTW::getNDims()
{
  if (dt == CPU)
    return mxGetNumberOfDimensions(pa_cpu);

#ifdef WITH_GPUARRAY
  if (dt == GPU) // always do this according to Matlab Doc
    return mxGPUGetNumberOfDimensions(pa_gpu);
#endif // WITH_GPUARRAY
}

mwSize xpuMxArrayTW::getSizeAtDim(mwSize dim)
{
  mwSize ndim = getNDims();
  if (dim >= ndim) return 1;

  if (dt == CPU)
    return mxGetDimensions(pa_cpu)[dim];

#ifdef WITH_GPUARRAY
  if (dt == GPU)
    return mxGPUGetDimensions(pa_gpu)[dim];
#endif // WITH_GPUARRAY
}

xpuMxArrayTW::DEV_TYPE xpuMxArrayTW::getDevice()
{
  return dt;
}

mxClassID xpuMxArrayTW::getElemType()
{
  if (dt == CPU)
    return mxGetClassID(pa_cpu);

#ifdef WITH_GPUARRAY
  return mxGPUGetClassID(pa_gpu);
#endif // WITH_GPUARRAY
}

void* xpuMxArrayTW::getDataBeg()
{
  if (dt == CPU)
    return mxGetData(pa_cpu);

#ifdef WITH_GPUARRAY
  return mxGPUGetData(pa_gpu);
#endif // WITH_GPUARRAY
}

void xpuMxArrayTW::setMxArray(mxArray *pa)
{
  pa_cpu = pa;
  dt = CPU;

#ifdef WITH_GPUARRAY
  if (mxIsGPUArray(pa)) {
    pa_gpu = (mxGPUArray*) mxGPUCreateFromMxArray(pa);
    dt = GPU;
  }
#endif // WITH_GPUARRAY
}

mxArray* xpuMxArrayTW::getMxArray()
{
  return pa_cpu;
}


//// Impl of shorthand
mxArray* createVol5d(mwSize sz[], xpuMxArrayTW::DEV_TYPE dt)
{
  if (dt == xpuMxArrayTW::CPU)
    return mxCreateNumericArray(5, sz, mxSINGLE_CLASS, mxREAL);

#ifdef WITH_GPUARRAY
  mxGPUArray* p = mxGPUCreateGPUArray(5, sz, mxSINGLE_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
  return mxGPUCreateMxArrayOnGPU(p);
#endif // WITH_GPUARRAY
}

mxArray* createVol5dZeros(mwSize sz[], xpuMxArrayTW::DEV_TYPE dt)
{
  if (dt == xpuMxArrayTW::CPU)
    return mxCreateNumericArray(5, sz, mxSINGLE_CLASS, mxREAL);

#ifdef WITH_GPUARRAY
  mxGPUArray* p = mxGPUCreateGPUArray(5, sz, mxSINGLE_CLASS, mxREAL, MX_GPU_INITIALIZE_VALUES);
  return mxGPUCreateMxArrayOnGPU(p);
#endif // WITH_GPUARRAY
}

mxArray* createVol5dLike(const xpuMxArrayTW &rhs, mxClassID tp /*= mxSINGLE_CLASS*/)
{
  // TODO: the right impl !
  return 0;
}


