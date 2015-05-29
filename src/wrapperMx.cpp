#include "wrapperMx.h"


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
    assert(pa_gpu != 0);
    mxGPUDestroyGPUArray(pa_gpu);
    pa_gpu = 0;
  }
#endif // WITH_GPUARRAY
}

xpuMxArrayTW::xpuMxArrayTW(const xpuMxArrayTW& rhs)
{
  // always do these stuff
  dt     = rhs.dt;
  pa_cpu = rhs.pa_cpu;
  pa_gpu = 0;

#ifdef WITH_GPUARRAY
  if ( rhs.dt == xpuMxArrayTW::GPU ) { // hold its own
    pa_gpu = (mxGPUArray*)mxGPUCreateFromMxArray(pa_cpu);
  }
#endif // WITH_GPUARRAY

  return;
}

xpuMxArrayTW& xpuMxArrayTW::operator=(const xpuMxArrayTW& rhs)
{
  // always do these stuff
  dt     = rhs.dt;
  pa_cpu = rhs.pa_cpu;
  pa_gpu = 0;

#ifdef WITH_GPUARRAY
  if ( rhs.dt == xpuMxArrayTW::GPU ) { // hold its own
    pa_gpu = (mxGPUArray*)mxGPUCreateFromMxArray(pa_cpu);
  }
#endif // WITH_GPUARRAY

  return *this;
}

mwSize xpuMxArrayTW::getNDims() const
{
#ifdef WITH_GPUARRAY
  if (dt == GPU) // always do this according to Matlab Doc
    return mxGPUGetNumberOfDimensions(pa_gpu);
#endif // WITH_GPUARRAY

  // else (dt == CPU)
  return mxGetNumberOfDimensions(pa_cpu);
}

mwSize xpuMxArrayTW::getSizeAtDim(mwSize dim) const
{
  mwSize ndim = getNDims();
  if (dim >= ndim) return 1;

#ifdef WITH_GPUARRAY
  if (dt == GPU)
    return (mxGPUGetDimensions(pa_gpu))[dim];
#endif // WITH_GPUARRAY

  // else (dt == CPU)
  return (mxGetDimensions(pa_cpu))[dim];
}

xpuMxArrayTW::DEV_TYPE xpuMxArrayTW::getDevice() const
{
  return dt;
}

mxClassID xpuMxArrayTW::getElemType() const
{
  if (dt == CPU)
    return mxGetClassID(pa_cpu);

#ifdef WITH_GPUARRAY
  return mxGPUGetClassID(pa_gpu);
#endif // WITH_GPUARRAY
}

void* xpuMxArrayTW::getDataBeg() const
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

mxArray* xpuMxArrayTW::getMxArray() const
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
  if (rhs.dt == xpuMxArrayTW::CPU)
    return mxCreateNumericArray(mxGetNumberOfDimensions(rhs.pa_cpu), 
                                mxGetDimensions(rhs.pa_cpu), 
                                tp, mxREAL);

#ifdef WITH_GPUARRAY
  mxGPUArray* p = mxGPUCreateGPUArray(mxGPUGetNumberOfDimensions(rhs.pa_gpu),
                                      mxGPUGetDimensions(rhs.pa_gpu),
                                      tp, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
  return mxGPUCreateMxArrayOnGPU(p);
#endif // WITH_GPUARRAY
}

mxArray* createVol5dZerosLike(const xpuMxArrayTW &rhs, mxClassID tp /*= mxSINGLE_CLASS*/)
{
  if (rhs.dt == xpuMxArrayTW::CPU)
    return mxCreateNumericArray(mxGetNumberOfDimensions(rhs.pa_cpu), 
    mxGetDimensions(rhs.pa_cpu), 
    tp, mxREAL); // 0s ensured

#ifdef WITH_GPUARRAY
  mxGPUArray* p = mxGPUCreateGPUArray(mxGPUGetNumberOfDimensions(rhs.pa_gpu),
    mxGPUGetDimensions(rhs.pa_gpu),
    tp, mxREAL, MX_GPU_INITIALIZE_VALUES); // with 0s
  return mxGPUCreateMxArrayOnGPU(p);
#endif // WITH_GPUARRAY
}


mwSize numel(const xpuMxArrayTW &rhs)
{
#ifdef WITH_GPUARRAY
  if (rhs.dt == xpuMxArrayTW::GPU)
    return mxGPUGetNumberOfElements(rhs.pa_gpu);
#endif // WITH_GPUARRAY

  // else (rhs.dt == xpuMxArrayTW::CPU)
  return mxGetNumberOfElements(rhs.pa_cpu);
}


