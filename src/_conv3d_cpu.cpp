#include "_conv3d_cpu.h"

conv3d_cpu::conv3d_cpu()
{

}

conv3d::CALL_TYPE conv3d_cpu::parse_and_set( int no, mxArray *vo[], int ni, mxArray const *vi[] )
{
  return conv3d::FPROP;
}

void conv3d_cpu::set_stride( mxArray const *pa )
{

}

void conv3d_cpu::set_pad( mxArray const *pa )
{

}

void conv3d_cpu::create_Y()
{

}

void conv3d_cpu::create_dX()
{

}

void conv3d_cpu::fprop()
{

}

void conv3d_cpu::bprop()
{

}