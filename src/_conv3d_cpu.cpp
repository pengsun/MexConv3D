#include "_conv3d_cpu.h"



conv3d_cpu::conv3d_cpu()
{

}

conv3d::CALL_TYPE conv3d_cpu::parse_and_set( int no, mxArray *vo[], int ni, mxArray const *vi[] )
{
  conv3d_cpu::CALL_TYPE ct;
  int n_opt = -1;

  // fprop or bprop?
  if (no == 1) {
    if ( ni < 3 || !mxIsSingle(vi[0]) || !mxIsSingle(vi[1]) || !mxIsSingle(vi[2]) ) 
      mexErrMsgTxt(THE_CMD);

    ct = FPROP;
    n_opt = 3;
    X = (mxArray*) vi[0]; // we won't change input!
    F = (mxArray*) vi[1];
    B = (mxArray*) vi[2];
  } else if (no == 1) {
    if ( ni < 4 ) 
      mexErrMsgTxt(THE_CMD);
    if ( !mxIsSingle(vi[0]) || !mxIsDouble(vi[1]) ||
         !mxIsSingle(vi[2]) || !mxIsDouble(vi[3]) )
      mexErrMsgTxt(THE_CMD);

    ct = BPROP;
    n_opt = 4;
    X  = (mxArray*) vi[0]; // we won't change input!
    F  = (mxArray*) vi[1];
    B  = (mxArray*) vi[2];
    dY = (mxArray*) vi[3];
  } else {
    mexErrMsgTxt(THE_CMD);
  }

  // parse option/value pairs
  if ( ((ni-n_opt)%2) != 0 ) // imbalance option/value
    mexErrMsgTxt(THE_CMD);
  for (int i = n_opt; i < ni; i+=2) {
    else if (isStrEqual(vi[i], "stride")) this->set_stride(vi[i+1]);
    else if (isStrEqual(vi[i], "pad"))    this->set_pad(vi[i+1]);
    else                                  mexErrMsgTxt(THE_CMD);
  } // for i

  return ct;
}

void conv3d_cpu::set_stride( mxArray const *pa )
{
  mexErrMsgTxt("Option stride not implemented yet. Sorry..."
    "Currently the stride is always 1.\n");
  //if ( !setCArray<mwSize, 3>(pa, this->stride) )
  //  mexErrMsgTxt(THE_CMD);
}

void conv3d_cpu::set_pad( mxArray const *pa )
{
  mexErrMsgTxt("Option pad not implemented yet. Sorry..."
    "Currently the pad is always 0.\n");
  //if ( !setCArray<mwSize, 6>(pa, this->pad) )
  //  mexErrMsgTxt(THE_CMD);
}

void conv3d_cpu::create_Y()
{
  // check
  if (getVolH(X) < getVolH(F) ||
      getVolW(X) < getVolW(F) ||
      getVolD(X) < getVolD(F) )
    mexErrMsgTxt(THE_CMD);

  // size Y TODO: the right size taking pad and stride into account
  mwSize HY = getVolH(X) - getVolH(F) + 1;
  mwSize WY = getVolW(X) - getVolW(F) + 1;
  mwSize DY = getVolD(X) - getVolD(F) + 1;
  mwSize MY = getVolQ(F);
  mwSize NY = getVolN(X);

  // create Y
  Y = createVol5d(HY, WY, DY, MY, NY);
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