#include "mxWrapper.h"
#include "_maxpool3d_gpu.h"


namespace {
  const float VERY_NEGATIVE_NUM = -1e20;
}

//// impl of public methods
maxpool3d_gpu::maxpool3d_gpu()
{

}

maxpool3d_gpu::maxpool3d_gpu(const maxpool3d &obj)
{
  for (int i = 0; i < 6; ++i) pad[i]  = obj.pad[i];
  for (int i = 0; i < 3; ++i) pool[i] = obj.pool[i];
  for (int i = 0; i < 3; ++i) stride[i] = obj.stride[i];

  ind = obj.ind;
  X  = obj.X;
  dX = obj.dX;
  Y  = obj.Y;
  dY = obj.dY;

  ct = obj.ct;

}

void maxpool3d_gpu::fprop()
{

}

void maxpool3d_gpu::bprop()
{
}

