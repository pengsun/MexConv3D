#include "_maxpool3d_cpu.h"

using maxpool3d::CALL_TYPE;

static const char* THE_CMD = 
  "Bad input or output arguments. The right way to call:"
  "[Y,ind] = MEX_MAXPOOL3D(X); forward pass"
  "dZdX = MEX_MAXPOOL3D(dZdY, ind); backward pass"
  "MEX_MAXPOOL3D(..., 'pool',pool, 'stride',s, 'pad',pad); options ";

CALL_TYPE maxpool3d_cpu::parse_and_set( int no, mxArray *vo[], int ni, mxArray const *vi[] )
{
  CALL_TYPE ct;
  int n_opt = -1;

  if (no == 2) {
    ct = FPROP;
    n_opt = 1;
    this->X = vi[0];
  } else if (no == 1) {
    ct = BPROP;
    n_opt = 2;
    this->dY  = vi[0];
    this->ind = vi[1];
  } else {
    mexErrMsgTxt(THE_CMD);
  }

  // parse option/value pairs
  if ( ((ni-n_opt)%2) != 0 ) // imbalance option/value
    mexErrMsgTxt(THE_CMD);
  for (int i = n_opt; i < ni; i+=2) {
    if      (isStrEqual(vi[i], "pool"))   this->set_pool(vi[i+1]);
    else if (isStrEqual(vi[i], "stride")) this->set_stride(vi[i+1]);
    else if (isStrEqual(vi[i], "pad"))    this->set_pad(vi[i+1]);
    else                                  mexErrMsgTxt(THE_CMD);
  } // for i

  return ct;
}

void maxpool3d_cpu::set_pool( mxArray const *pa )
{
  if ( !setCArray<mwSize, 3>(pa, this->pool) )
    mexErrMsgTxt(THE_CMD);
}

void maxpool3d_cpu::set_stride( mxArray const *pa )
{
  if ( !setCArray<mwSize, 3>(pa, this->stride) )
    mexErrMsgTxt(THE_CMD);
}

void maxpool3d_cpu::set_pad( mxArray const *pa )
{
  if ( !setCArray<mwSize, 6>(pa, this->pad) )
    mexErrMsgTxt(THE_CMD);
}

void maxpool3d_cpu::fprop()
{
  // input X
  mwSize xH = getVolH(X);
  mwSize xHW = getVolH(X) * getVolW(X);

  // output Y
  float* yy = (float*)mxGetData(Y);
  for (mwSize n = 0; n < numVol(Y); ++n) { // Y dim4,dim5...: along each volume
    float * x_n = getVolDataBeg(X, n);

    for (mwSize k = 0; k < getVolD(Y); ++k) {     // Y dim3: along depth
      for (mwSize j = 0; j < getVolW(Y); ++j) {   // Y dim2: along width
        for (mwSize i = 0; i < getVolH(Y); ++i) { // Y dim1: along height
          // init value for current Y
          float vmax = -1e6;

          // set the window on X for current Y element (yy)
          mwSize offset[3];
          offset[0] = i*pool[0]; 
          offset[1] = j*pool[1];
          offset[2] = k*pool[2];
          float *xwin = x_n + offset[0] + offset[1]*xH + offset[2]*xHW;
          
          // inspect the window at X, get the max value
          for (mwSize t = 0; t < pool[2]; ++t) {     // X window dim3
            for (mwSize s = 0; s < pool[1]; ++s) {   // X window dim2
              for (mwSize r = 0; r < pool[0]; ++r) { // X window dim1
                // current x value
                float vx = *(xwin + r + s*xH + t*xHW);
                vmax = std::max(vx, vmax);
              } // r
            } // s
          } // t
          
          // write back to Y and advance
          *yy = vmax;
          ++yy;
        } // i
      } // j
    } // k
    
  } // n

}




