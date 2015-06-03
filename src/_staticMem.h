#pragma once
#include "wrapperMx.h"

// static memory across mex calling, all zeros
void* sm_zeros_cpu (size_t nelem, mxClassID et);

// static memory across mex calling, all ones
void* sm_ones_cpu (size_t nelem, mxClassID et);

// release all (called typically when unloading mex file)
void* sm_release_cpu ();


// static memory across mex calling, all zeros
void* sm_zeros_gpu (size_t nelem, mxClassID et);

// static memory across mex calling, all ones
void* sm_ones_gpu (size_t nelem, mxClassID et);

// release all (called typically when unloading mex file)
void* sm_release_gpu ();