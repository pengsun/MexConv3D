#pragma once

#include <cassert>
#include "_staticMem.h"
#ifdef WITH_GPUARRAY
#include "_cu_helper.cuh"
#endif // WITH_GPUARRAY


namespace {
  static buf_cpu_t bufZeros_cpu;
  static buf_cpu_t bufOnes_cpu;

#ifdef WITH_GPUARRAY
  static buf_gpu_t bufZeros_gpu;
  static buf_gpu_t bufOnes_gpu;
#endif // WITH_GPUARRAY
}


//// Impl of buf_t
buf_t::buf_t()
: beg(0), nelem(0)
{

}

bool buf_t::is_need_realloc( size_t _nelem)
{
  return ( beg==0 || _nelem>nelem );
}


//// Impl of buf_cpu_t
void buf_cpu_t::realloc( size_t _nelem )
{
  dealloc();
  beg = (float*) malloc( _nelem * sizeof(float) );
  if (beg == 0) throw sm_ex("staticMem: Out of CPU memory.\n");
}

void buf_cpu_t::dealloc()
{
  if (beg != 0) free( (void*) beg );
  beg = 0;
}

#ifdef WITH_GPUARRAY
//// Impl of buf_gpu_t
void buf_gpu_t::realloc( size_t _nelem )
{
  dealloc();
  beg = (float*) malloc( _nelem * sizeof(float) );
  if ( cudaSuccess != cudaMalloc((void*)&beg, nelem*sizeof(float)) )
    throw sm_ex("staticMem: Out of GPU memory.\n");
}

void buf_gpu_t::dealloc()
{
  if (beg != 0) cudaFree( (void*) beg );
  beg = 0;
}
#endif

//// Impl of the public interface
float* sm_zeros (size_t nelem, xpuMxArrayTW::DEV_TYPE dt)
{
#ifdef WITH_GPUARRAY
  if (dt == xpuMxArrayTW::GPU) {
    if ( bufZeros_gpu.is_need_realloc(nelem) ) {
      bufZeros_gpu.realloc(nelem);

      // set initial value
      dim3 sz_blk( ceil_divide(nelem,CU_NUM_THREADS));
      dim3 sz_thd(CU_NUM_THREADS );
      kernelSetZero<float><<<sz_blk,sz_thd>>>(bufZeros_gpu.beg, nelem);
    }
    return bufZeros_gpu.beg;
  }
#endif // WITH_GPUARRAY
  
  assert(dt == xpuMxArrayTW::CPU);

  if ( bufZeros_cpu.is_need_realloc(nelem) ) {
   bufZeros_cpu.realloc(nelem);
   for (int i = 0; i < nelem; ++i) bufZeros_cpu.beg[i] = 0.0;
  }
  return bufZeros_cpu.beg;
}

float* sm_ones (size_t nelem, xpuMxArrayTW::DEV_TYPE dt)
{
#ifdef WITH_GPUARRAY
  if (dt == xpuMxArrayTW::GPU) {
    if ( bufOnes_gpu.is_need_realloc(nelem) ) {
      bufOnes_gpu.realloc(nelem);

      // set initial value
      dim3 sz_blk( ceil_divide(nelem,CU_NUM_THREADS));
      dim3 sz_thd(CU_NUM_THREADS );
      kernelSetOne<float><<<sz_blk,sz_thd>>>(bufOnes_gpu.beg, nelem);
    }
    return bufZeros_gpu.beg;
  }
#endif // WITH_GPUARRAY

  assert(dt == xpuMxArrayTW::CPU);

  if ( bufOnes_cpu.is_need_realloc(nelem) ) {
    bufOnes_cpu.realloc(nelem);
    for (int i = 0; i < nelem; ++i) bufOnes_cpu.beg[i] = 1.0;
  }
  return bufOnes_cpu.beg;
}

void sm_release ()
{
  bufZeros_cpu.dealloc();
  bufOnes_cpu.dealloc();

#ifdef WITH_GPUARRAY
  bufZeros_gpu.dealloc();
  bufOnes_gpu.dealloc();
#endif // WITH_GPUARRAY
}