#include "mex_shorthand.h"
#include "_maxpool3d_cpu.h"
#include <omp.h>

maxpool3d_cpu::maxpool3d_cpu()
{

}

maxpool3d::CALL_TYPE maxpool3d_cpu::parse_and_set( int no, mxArray *vo[], int ni, mxArray const *vi[] )
{
  maxpool3d::CALL_TYPE ct;
  int n_opt = -1;

  // fprop or bprop?
  if (no == 2) {
    if ( ni < 1 || !mxIsSingle(vi[0]) ) 
      mexErrMsgTxt(THE_CMD);

    ct = FPROP;
    n_opt = 1;
    this->X = (mxArray*) vi[0]; // we won't change input!
  } else if (no == 1) {
    if ( ni < 2 || !mxIsSingle(vi[0]) || !mxIsDouble(vi[1]) ) 
      mexErrMsgTxt(THE_CMD);

    ct = BPROP;
    n_opt = 2;
    this->dY  = (mxArray*) vi[0]; // we won't change input!
    this->ind = (mxArray*) vi[1];
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


void maxpool3d_cpu::create_Y()
{
  // header: X 
  mwSize ndimX = mxGetNumberOfDimensions(X);
  if (ndimX < 3) mexErrMsgTxt(THE_CMD);
  const mwSize* szX  = mxGetDimensions(X);
  if (!mxIsSingle(X)) mexErrMsgTxt(THE_CMD);

  // header: Y TODO the right size!!
  mwSize szY[5] = {0,0,0,1,1};
  for (int i = 0; i < 3; ++i)
    szY[i] = szX[i] / pool[i];
  for (int i = 3; i < ndimX; ++i)
    szY[i] = szX[i];

  // create Y
  this->Y = mxCreateNumericArray(5, szY, mxSINGLE_CLASS, mxREAL);
}

void maxpool3d_cpu::create_ind()
{
  ind = mxCreateNumericArray(mxGetNumberOfDimensions(Y), mxGetDimensions(Y),
                             mxDOUBLE_CLASS, mxREAL);
}


void maxpool3d_cpu::create_dX()
{
  // check ind
  if (!mxIsDouble(ind)) mexErrMsgTxt(THE_CMD);

  // header: dY
  mwSize ndimY = mxGetNumberOfDimensions(dY);
  if (ndimY < 3) mexErrMsgTxt(THE_CMD);
  const mwSize* szY  = mxGetDimensions(dY);
  if (!mxIsSingle(dY)) mexErrMsgTxt(THE_CMD);

  // header: dX TODO the right size!!
  mwSize szX[5] = {0,0,0,1,1};
  for (int i = 0; i < 3; ++i)
    szX[i] = szY[i] * pool[i];
  for (int i = 3; i < ndimY; ++i)
    szX[i] = szY[i];

  // create dX
  this->dX = mxCreateNumericArray(5, szX, mxSINGLE_CLASS, mxREAL);

  // make sure they are zeros
  mwSize num = mxGetNumberOfElements(dX);
  float* ptr = (float*)mxGetData(dX);
  for (mwSize n = 0; n < num; ++n, ++ptr) *ptr = 0.0;
}

void maxpool3d_cpu::fprop()
{
  // input X size
  mwSize xH   = getVolH(X);
  mwSize xHW  = getVolH(X) * getVolW(X);
  mwSize xHWD = numelVol(X);
  // create output
  this->create_Y();
  this->create_ind();

  // iterate over Y, record the max value and index
  #pragma omp parallel for
  for (int64_T n = 0; n < numVol(Y); ++n) { // Y dim4,dim5...: along each volume
    float* const xx = getVolDataBeg<float>(X, n); // input port (never change it)
    float* yy  = getVolDataBeg<float>(Y, n);      // output port 
    double* ii = getVolDataBeg<double>(ind, n);

    for (mwSize k = 0; k < getVolD(Y); ++k) {     // Y dim3: along depth
      for (mwSize j = 0; j < getVolW(Y); ++j) {   // Y dim2: along width
        for (mwSize i = 0; i < getVolH(Y); ++i) { // Y dim1: along height
          // init value for current Y
          float vmax = -1e6;
          double imax = -1.0;

          // set the window on X for current Y element (yy)
          mwSize offset[3];
          offset[0] = i*pool[0]; 
          offset[1] = j*pool[1];
          offset[2] = k*pool[2];
          float *xwin = xx + offset[0] + offset[1]*xH + offset[2]*xHW;
          
          // inspect the window at X, get the max value
          for (mwSize t = 0; t < pool[2]; ++t) {     // X window dim3
            for (mwSize s = 0; s < pool[1]; ++s) {   // X window dim2
              for (mwSize r = 0; r < pool[0]; ++r) { // X window dim1
                // current x value
                float vx = *(xwin + r + s*xH + t*xHW);
                // found new max value?
                if (vx > vmax) {
                  vmax = vx;
                  imax = double( (offset[0]+r) + (offset[1]+s)*xH + (offset[2]+t)*xHW + n*xHWD );
                }
              } // r
            } // s
          } // t
          
          // write back to Y and advance
          *yy++ = vmax;
          *ii++ = imax + 1; // to matlab 1-base
        } // i
      } // j
    } // k
    
  } // parallel for n

}

void maxpool3d_cpu::bprop()
{
  // create dX at input port
  this->create_dX();

  // dX at input port
  float* const dxx = (float*)mxGetData(this->dX);
  // dY and index at output port
  float* const dyy = (float*)mxGetData(this->dY);
  double* const ii = (double*)mxGetData(this->ind);
  
  // iterate over dY, set dX
  mwSize num = mxGetNumberOfElements(this->dY);
  #pragma omp parallel for
  for (int64_T n = 0; n < num; ++n) {
    mwSize ix = mwSize( ii[n] );
    ix -= 1; // matlab 1-base -> C++ 0-base
    *(dxx + ix) = dyy[n];
  }
}
