#pragma once
#include "_staticMem.h"


void* sm_zeros (size_t nelem, mxClassID et, xpuMxArrayTW::DEV_TYPE dt)
{
  if (dt == xpuMxArrayTW::DEV_TYPE::CPU)
    return sm_zeros_cpu(nelem, et);

#ifdef WITH_GPUARRAY
  return sm_zeros_gpu(nelem, et);
#endif // WITH_GPUARRAY

}

void* sm_ones (size_t nelem, mxClassID et, xpuMxArrayTW::DEV_TYPE dt)
{
  if (dt == xpuMxArrayTW::DEV_TYPE::CPU)
    return sm_zeros_cpu(nelem, et);

#ifdef WITH_GPUARRAY
  return sm_zeros_gpu(nelem, et);
#endif // WITH_GPUARRAY
}

// release all (called typically when unloading mex file)
void* sm_release ()
{
  sm_release_cpu();

#ifdef WITH_GPUARRAY
  sm_release_gpu();
#endif // WITH_GPUARRAY
}