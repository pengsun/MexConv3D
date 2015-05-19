#include "mex.h"
#include "src/conv3d.h"
#include "mex_shorthand.h"

namespace {

conv3d*      h = 0;
void cleanup ()
{
  safe_delete(h);
}

} // namespace


void mexFunction(int no, mxArray       *vo[],
                 int ni, mxArray const *vi[])
{
  // init resource
  mexAtExit(cleanup);
#ifdef WITHCUDNN
  factory_c3d_withcudnn factory;
#else
  factory_c3d_homebrew factory;
#endif
  cleanup();
  h = factory.create(vi[0],vi[1],vi[2]); // always expect X, F, B 

  // parse input
  conv3d::CALL_TYPE ct = h->parse_and_set(no,vo,ni,vi);

  // do the job and set output
  if (ct == conv3d::FPROP) {
    h->fprop();
    vo[0] = h->Y;
  }
  if (ct == conv3d::BPROP) {
    h->bprop();
    vo[0] = h->dX;
    vo[1] = h->dF;
    vo[2] = h->dB;
  }

  // on leave
  cleanup();
  return;
}