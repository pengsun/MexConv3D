#include "mex.h"
#include "src/maxpool3d.h"


static factory_mp3d* factory = 0;
static maxpool3d*    h = 0;
static void cleanup ()
{
  safe_delete(factory);
  safe_delete(h);
}

//
// [Y,ind] = MEX_MAXPOOL3D(X); forward pass
// dZdX = MEX_MAXPOOL3D(dZdY, ind); backward pass
// MEX_MAXPOOL3D(..., 'pool',pool, 'stride',s, 'pad',pad); options
//
void mexFunction(int no, mxArray       *vo[],
                 int ni, mxArray const *vi[])
{
  // init resource
  mexAtExit(cleanup);
  assert(factory==0);
#ifdef WITHCUDNN
  factory = new factory_mp3d_withcudnn;
#else
  factory = new factory_mp3d_homebrew;
#endif
  assert(h==0);
  h = factory->create(vi[0]);

  // parse input
  maxpool3d::CALL_TYPE ct = h->parse_and_set(no,vo,ni,vi);

  // do the job and set output
  if (ct == maxpool3d::FPROP) {
    h->fprop();
    vo[0] = h->Y;
    vo[1] = h->ind;
  }
  if (ct == maxpool3d::BPROP) {
    h->bprop();
    vo[0] = h->dX;
  }

  // on leave
  cleanup();
  return;
}