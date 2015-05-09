#include "maxpool3d.h"
#include "_maxpool3d_cpu.h"


maxpool3d* factory_mp3d_homebrew::create( mxArray const *from )
{
  return new maxpool3d_cpu;
}
