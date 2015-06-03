#pragma once
#include "wrapperMx.h"

// static memory across mex calling, all zeros
void* sm_zeros (size_t nelem, mxClassID et, xpuMxArrayTW::DEV_TYPE dt);

// static memory across mex calling, all ones
void* sm_ones (size_t nelem, mxClassID et, xpuMxArrayTW::DEV_TYPE dt);

// release all (called typically when unloading mex file)
void* sm_release ();