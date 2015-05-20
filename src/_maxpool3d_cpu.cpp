#include "mex_shorthand.h"
#include "_maxpool3d_cpu.h"
#include <omp.h>


namespace {
  const float VERY_NEGATIVE_NUM = -1e20;
}

//// impl of public methods
maxpool3d_cpu::maxpool3d_cpu()
{

}

void maxpool3d_cpu::fprop()
{
  // create output
  this->create_Y();
  this->create_ind();

  // input X size
  mwSize xH = getVolH(X), xW = getVolW(X), xD = getVolD(X);
  mwSize xHW  = xH*xW;
  mwSize xHWD = xH*xW*xD;

  // iterate over Y, record the max value and index
  #pragma omp parallel for
  for (int64_T n = 0; n < numVol(Y); ++n) { // Y dim4, dim5...: along each volume

    float*  yy = getVolDataBeg<float>(Y, n);    // output data
    double* ii = getVolDataBeg<double>(ind, n); // output index

    const float* const xx_beg = getVolDataBeg<float>(X, n); // input data (never change it)


    for (mwSize k = 0; k < getVolD(Y); ++k) {     // Y dim3: along depth
      for (mwSize j = 0; j < getVolW(Y); ++j) {   // Y dim2: along width
        for (mwSize i = 0; i < getVolH(Y); ++i) { // Y dim1: along height

          // init value for current Y
          float  vmax = VERY_NEGATIVE_NUM;
          double imax = -43.0;

          // set the window on X for current Y element (yy): the offset can be negative
          int64_T xwin_offset[3];
          xwin_offset[0] = -static_cast<int64_T>(pad[0]) + static_cast<int64_T>( i*stride[0] ); 
          xwin_offset[1] = -static_cast<int64_T>(pad[2]) + static_cast<int64_T>( j*stride[1] );
          xwin_offset[2] = -static_cast<int64_T>(pad[4]) + static_cast<int64_T>( k*stride[2] );
          const float* const xwin_beg = xx_beg + 
                                        xwin_offset[0] + 
                                        xwin_offset[1]*xH + 
                                        xwin_offset[2]*xHW;

          // inspect the window at X, get the max value
          for (int64_T t = 0; t < pool[2]; ++t) {     // X window dim3: depth
            int64_T xt = t + xwin_offset[2];
            bool xtInRange = (xt>=0) && (xt<xD);

            for (int64_T s = 0; s < pool[1]; ++s) {   // X window dim2: width
              int64_T xs = s + xwin_offset[1];
              bool xsInRange = (xs>=0) && (xs<xW);

              for (int64_T r = 0; r < pool[0]; ++r) { // X window dim1: height
                int64_T xr = r + xwin_offset[0];
                bool xrInRange = (xr>=0) && (xr<xH);

                // if out of range: never collect the element
                if ( !(xtInRange && xsInRange && xrInRange) )
                  continue;

                // collect the element: current x value
                float vx = *(xwin_beg + r + s*xH + t*xHW);
                if (vx >= vmax) { // found new max value?
                  vmax = vx;
                  imax = double( xr + xs*xH + xt*xHW + n*xHWD );
                } // if

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
  
  //#pragma omp parallel for
  for (int64_T n = 0; n < num; ++n) {
    mwSize ix = mwSize( ii[n] );
    ix -= 1; // matlab 1-base -> C++ 0-base

    // accumulate! there can be overlapping ix!
    dxx[ix] += dyy[n];
  }
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

//// impl of helpers
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
  // check dimensions: should we?
  //mwSize ndimX = mxGetNumberOfDimensions(X);
  //if (ndimX < 3) mexErrMsgTxt(THE_CMD);

  //
  check_pad_pool();

  // pooling window size should not be greater than feature map size
  if (pad[0]+pad[1]+getVolH(X) < pool[0] || 
      pad[2]+pad[3]+getVolW(X) < pool[1] ||
      pad[4]+pad[5]+getVolD(X) < pool[2] )
    mexErrMsgTxt(THE_CMD);

  // size Y: the right size taking into account pad and stride
  mwSize HY = (pad[0]+getVolH(X)+pad[1] - pool[0])/stride[0] + 1;
  mwSize WY = (pad[2]+getVolW(X)+pad[3] - pool[1])/stride[1] + 1;
  mwSize DY = (pad[4]+getVolD(X)+pad[5] - pool[2])/stride[2] + 1;
  mwSize MY = getSzAtDim<4>(X);
  mwSize NY = getSzAtDim<5>(X);

  // create Y
  Y = createVol5d(HY, WY, DY, MY, NY);
}

void maxpool3d_cpu::create_ind()
{
  ind = createVol5dLike(Y, mxDOUBLE_CLASS);
}

void maxpool3d_cpu::create_dX()
{
  // check ind & dY
  if (!mxIsDouble(ind) || !mxIsSingle(dY)) 
    mexErrMsgTxt(THE_CMD);

  //
  check_pad_pool();

  // size dX: the right size taking into account pad and stride
  mwSize szdX[5] = {0,0,0,1,1};
  szdX[0] = stride[0]*(getVolH(dY)-1) - (pad[0]+pad[1]) + pool[0];
  szdX[1] = stride[1]*(getVolW(dY)-1) - (pad[2]+pad[3]) + pool[1];
  szdX[2] = stride[2]*(getVolD(dY)-1) - (pad[4]+pad[5]) + pool[2];
  szdX[3] = getSzAtDim<4>(dY);
  szdX[4] = getSzAtDim<5>(dY);

  // create Y
  dX = createVol5dZeros(szdX);
}

void maxpool3d_cpu::check_pad_pool()
{
  if ( (pad[0]+pad[1]) >= pool[0] ||
       (pad[2]+pad[3]) >= pool[1] ||
       (pad[4]+pad[5]) >= pool[2] )
    mexErrMsgTxt(THE_CMD);
}