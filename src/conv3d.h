#pragma once
#include "mex.h"

//// the transformer
struct conv3d {
  conv3d ();

  // options
  mwSize pad[6];
  mwSize stride[3];
  // intermediate data: filter and bias
  mxArray *F, *dF;
  mxArray *B, *dB;
  // data at input/output port
  mxArray *X, *dX;
  mxArray *Y, *dY;

  // forward/backward propagation
  virtual void fprop () = 0;
  virtual void bprop () = 0;

  // helper: command, parser
  static const char * THE_CMD;
  enum CALL_TYPE {FPROP, BPROP};
  virtual CALL_TYPE parse_and_set (int no, mxArray *vo[], int ni, mxArray const *vi[]) = 0;
};


//// factory
struct factory_c3d {
  virtual conv3d* create (mxArray const *X, mxArray const *F, mxArray const *B, 
                          mxArray const *dY = 0) = 0;
};

struct factory_c3d_homebrew : public factory_c3d {
  conv3d* create (mxArray const *X, mxArray const *F, mxArray const *B, 
                  mxArray const *dY = 0);
};

struct factory_c3d_withcudnn : public factory_c3d { 
  // 3D data not implemented in cudnn yet...could be the case in the future?
  conv3d* create (mxArray const *X, mxArray const *F, mxArray const *B, 
                  mxArray const *dY = 0);
};