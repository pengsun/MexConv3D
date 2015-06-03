#pragma once
#include "_staticMem.h"

namespace {

static size_t    cur_nelem = 0;
static mxClassID cur_et = mxSINGLE_CLASS;
static void*     cur_beg = 0;

void* alloc (size_t nelem, mxClassID et) {

  return 0;
}

}

void* sm_zeros_cpu (size_t nelem, mxClassID et)
{
  if ()

}

void* sm_ones_cpu (size_t nelem, mxClassID et)
{

}

// release all (called typically when unloading mex file)
void* sm_release_cpu ()
{

}