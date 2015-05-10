#pragma once
#include "mex.h"

//// the transformer
struct maxpool3d {
  maxpool3d ();

  // options
  mwSize pad[6];
  mwSize pool[3];
  mwSize stride[3];
  // intermediate data: max elements index
  mxArray *ind;
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
struct factory_mp3d {
  virtual maxpool3d* create (mxArray const *from) = 0;
};

struct factory_mp3d_homebrew : public factory_mp3d {
  maxpool3d* create (mxArray const *from);
};

struct factory_mp3d_withcudnn : public factory_mp3d { 
  // 3D data not implemented in cudnn yet...could be the case in the future?
  maxpool3d* create (mxArray const *from);
};