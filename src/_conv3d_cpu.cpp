#include "_conv3d_cpu.h"


conv3d_cpu::conv3d_cpu()
{

}

void conv3d_cpu::fprop()
{
  create_Y();
  init_convmat();
  init_u(); 

  // iterate over each training instance
  mwSize N = getVolN(X);
  for (mwSize i = 0; i < N; i++) {
    // make phiX: the convolution matrix
    vol_to_convmat(X, i);

    // convolution: Y_ = phiX * F_
    matw F_ = make_F_();
    matw Y_ = make_Y_(i);
    CeqAxB(convmat, F_, Y_);

    // plus the bias: Y_ += u * B
    matw B_ = make_B_();
    CeqAxB(u, B_, Y_);
  }

  free_u();
  free_convmat();
}

void conv3d_cpu::bprop()
{
  create_dX();
  create_dF();
  create_dB();
  init_convmat();
  init_u();

  mwSize N = getVolN(X);
  matw dF_ = make_dF_();
  matw dB_ = make_dB_();
  for (mwSize i = 0; i < N; ++i) {
    // make phiX: the convolution matrix
    vol_to_convmat(X, i);

    // dF += phiX' * dY_
    matw dY_ = make_dY_(i);
    CeqATxB(convmat, dY_, dF_);

    // dB += u' * dY
    CeqATxB(u, dY_, dB_);

    // dphiX = dY * F'
    bool overwrite = true; // safe to reuse phiX memory! but remember to overwrite it!
    matw F_ = make_F_();
    CeqAxBT(dY_, F_, convmat, overwrite);
    // dX(:,:,:,:,i) <-- dphiX
    convmat_to_vol(dX, i);
  }

  free_u();
  free_convmat();
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
    X = (mxArray*) vi[0]; // we hereby guarantee that we won't change the inputs!
    F = (mxArray*) vi[1];
    B = (mxArray*) vi[2];
  } 
  else if (no == 3) {
    if ( ni < 4 ) 
      mexErrMsgTxt(THE_CMD);
    if ( !mxIsSingle(vi[0]) || !mxIsSingle(vi[1]) ||
         !mxIsSingle(vi[2]) || !mxIsSingle(vi[3]) )
      mexErrMsgTxt(THE_CMD);

    ct = BPROP;
    n_opt = 4;
    X  = (mxArray*) vi[0]; // we hereby guarantee that we won't change the inputs!
    F  = (mxArray*) vi[1];
    B  = (mxArray*) vi[2];
    dY = (mxArray*) vi[3];
  } 
  else {
    mexErrMsgTxt(THE_CMD);
  }

  // parse option/value pairs
  if ( ((ni-n_opt)%2) != 0 ) // imbalance option/value
    mexErrMsgTxt(THE_CMD);
  for (int i = n_opt; i < ni; i+=2) {
    if (isStrEqual(vi[i], "stride"))   this->set_stride(vi[i+1]);
    else if (isStrEqual(vi[i], "pad")) this->set_pad(vi[i+1]);
    else                               mexErrMsgTxt(THE_CMD);
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
  // check input X and filter F, B
  if (getVolP(F) != getVolM(X))  // #feature maps should match
    mexErrMsgTxt(THE_CMD);

  if (getVolQ(F) != mxGetNumberOfElements(B)) // #Bias should math the output
    mexErrMsgTxt(THE_CMD);

  if (getVolH(X) < getVolH(F) || // filter should not be greater than feature map
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

matw conv3d_cpu::make_F_()
{
  matw F_;
  F_.beg = getDataBeg<float>(F);
  F_.H   = numelVol(F) * getVolP(F);
  F_.W   = getVolQ(F);

  return F_;
}

matw conv3d_cpu::make_Y_(mwSize i)
{
  matw Y_;
  Y_.beg = getVolInstDataBeg<float>(Y, i);
  Y_.H   = numelVol(Y);
  Y_.W   = getSzAtDim<4>(Y);

  return Y_;
}

matw conv3d_cpu::make_B_()
{
  matw B_;
  B_.beg = getDataBeg<float>(B);
  B_.H   = 1;
  B_.W   = mxGetNumberOfElements(B);

  return B_;
}

void conv3d_cpu::create_dX()
{
  dX = createVol5dLike(X);
}

void conv3d_cpu::create_dF()
{
  dF = createVol5dLike(F);
}

void conv3d_cpu::create_dB()
{
  dB = createVol5dLike(B);
}

matw conv3d_cpu::make_dX_(mwSize i)
{
  matw dX_;
  dX_.beg = getVolInstDataBeg<float>(dX, i);
  dX_.H   = numelVol(dX);
  dX_.W   = getSzAtDim<4>(dX);

  return dX_;
}

matw conv3d_cpu::make_dY_(mwSize i)
{
  matw dY_;
  dY_.beg = getVolInstDataBeg<float>(dY, i);
  dY_.H   = numelVol(dY);
  dY_.W   = getSzAtDim<4>(dY);

  return dY_;
}

matw conv3d_cpu::make_dF_()
{
  matw dF_;
  dF_.beg = getVolDataBeg<float>(dF);
  dF_.H   = numelVol(dF) * getSzAtDim<4>(dF);
  dF_.W   = getSzAtDim<5>(dF);

  return dF_;
}

matw conv3d_cpu::make_dB_()
{
  matw dB_;
  dB_.beg = getVolDataBeg<float>(dB);
  dB_.H   = 1;
  dB_.W   = mxGetNumberOfElements(dB);
  
  return dB_;
}

void conv3d_cpu::init_convmat()
{
  convmat.H = numelVol(Y);
  convmat.W = numelVol(F) * getVolP(F);
  mwSize nelem = convmat.H * convmat.W;
  convmat.beg = (float*)mxCalloc( nelem, sizeof(float) );
  // mxCalloc assures the initialization with all 0s ! 
}

void conv3d_cpu::free_convmat()
{
  mxFree( (void*)convmat.beg );
}

void conv3d_cpu::vol_to_convmat(const mxArray *pvol, mwSize i)
{

}

void conv3d_cpu::convmat_to_vol(mxArray *pvol, mwSize i)
{

}

void conv3d_cpu::init_u()
{
  u.H = numelVol(Y);
  u.W = 1;
  mwSize nelem = u.H * u.W ;
  u.beg = (float*)mxMalloc( nelem * sizeof(float) );

  // make sure all one
  for (int i = 0; i < nelem; i++)
    u.beg[i] = 1.0;
}

void conv3d_cpu::free_u()
{
  mxFree( (void*)u.beg );
}