#include "mex.h"
#include "src/conv3d.h"
#include "mex_shorthand.h"

// "Y = MEX_CONV3D(X,F,B); forward pass"
// "[dX,dF,dB] = MEX_CONV3D(X,F,B, dY); backward pass"
// "MEX_CONV3D(..., 'stride',s, 'pad',pad); options"
void mexFunction(int no, mxArray       *vo[],
                 int ni, mxArray const *vi[])
{
#ifdef WITHCUDNN
  factory_c3d_withcudnn factory;
#else
  factory_c3d_homebrew factory;
#endif

  conv3d* h = 0; // TODO: consider unique_ptr?
  try {
    h = factory.create(vi[0],vi[1],vi[2]); // always expect X, F, B 

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
    // done: cleanup
    safe_delete(h);

  } 
  catch (const conv3d_ex& e) {
    safe_delete(h);
    mexErrMsgTxt( e.what() );
  } 

}