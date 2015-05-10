#include "maxpool3d.h"
#include "_maxpool3d_cpu.h"

//// Impl of maxpool3d
const char* maxpool3d::THE_CMD = 
  "Bad input or output arguments. The right way to call:\n"
  "[Y,ind] = MEX_MAXPOOL3D(X); forward pass\n"
  "dZdX = MEX_MAXPOOL3D(dZdY, ind); backward pass\n"
  "MEX_MAXPOOL3D(..., 'pool',pool, 'stride',s, 'pad',pad); options\n"
  "X, dZdY must be single, ind must be double;\n";

maxpool3d::maxpool3d()
{
  pool[0] = pool[1] = pool[2] = 2;
  stride[0] = stride[1] = stride[2] = 2;
  pad[0] = pad[1] = pad[2] = pad[3] = pad[4] = pad[5] = 0;
}


//// Impl of factory
maxpool3d* factory_mp3d_homebrew::create( mxArray const *from )
{
  return new maxpool3d_cpu;
}


