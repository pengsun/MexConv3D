#pragma once
#include "wrapperMx.h"

//// buffer 
struct buf_t {
  buf_t ();
  
  virtual void realloc (size_t _nelem) = 0;
  virtual void dealloc () = 0;

  bool is_need_realloc (size_t _nelem);
  
  float* beg; // host or device raw data pointer
  int nelem; // #bytes = nelem * sizeof(et)
};


//// CPU buffer
struct buf_cpu_t : public buf_t {
  void realloc (size_t _nelem);
  void dealloc ();
};

//// GPU buffer
struct buf_gpu_t : public buf_t {
  void realloc (size_t _nelem);
  void dealloc ();
};