#include "conv3d.h"
#include "_conv3d_cpu.h"

const char* conv3d::THE_CMD = 
  "Bad input or output arguments. The right way to call:\n"
  "Y = MEX_CONV3D(X,F,B); forward pass\n"
  "[dX,dF,dB] = MEX_CONV3D(X,F,B, dY); backward pass\n"
  "MEX_CONV3D(..., 'stride',s, 'pad',pad); options\n"
  "All arguments must be single.\n";

conv3d::conv3d()
{
  stride[0] = stride[1] = stride[2] = 2;
  pad[0] = pad[1] = pad[2] = pad[3] = pad[4] = pad[5] = 0;

  F = dF = 0;
  B = dB = 0;
  X = dX = 0;
  Y = dY = 0;
}

conv3d* factory_c3d_homebrew::create(
  mxArray const *X, mxArray const *F, mxArray const *B, mxArray const *dY)
{
  if (!mxIsSingle(X) || !mxIsSingle(F) || !mxIsSingle(B))
    mexErrMsgTxt(conv3d::THE_CMD);

  return new conv3d_cpu;
}