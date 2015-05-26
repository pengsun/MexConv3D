#include "_conv3d_blas_cpu.h"

namespace {

// helper: sub volume attaching to big volume
struct subvol4D {
  float* beg;
  int64_T offset[4];
  mwSize  sizeBigVol[4];

  mwSize size[4];
  mwSize stride[4];

  void copy_to_row           (matw& to,   mwSize row);
  void copy_and_inc_from_row (matw& from, mwSize row);
};

void subvol4D::copy_to_row (matw& the_mat, mwSize row)
{
  // init output: ptr element matrix
  float* pe_mat = the_mat.beg + row; 
  
  // For the sub volume, scan dim3,...,dim0. At each dim,
  // consider three cases: underflow, regular walking and overflow
  for (mwSize d3 = 0; d3 < size[3]; ++d3) {
    int64_T d3BigVol = (offset[3] + d3);
    float* ptr_d3    = d3BigVol*stride[3] + this->beg;
    bool   d3InRange = (d3BigVol >= 0) && (d3BigVol < sizeBigVol[3]);

    for (mwSize d2 = 0; d2 < size[2]; ++d2) {
      int64_T d2BigVol = (offset[2] + d2);
      float* ptr_d2    = d2BigVol*stride[2] + ptr_d3;
      bool   d2InRange = (d2BigVol >= 0) && (d2BigVol < sizeBigVol[2]); 

      for (mwSize d1 = 0; d1 < size[1]; ++d1) {
        int64_T d1BigVol = (offset[1] + d1);
        float* ptr_d1    = d1BigVol*stride[1] + ptr_d2;
        bool   d1InRange = (d1BigVol >= 0) && (d1BigVol < sizeBigVol[1]);

        for (mwSize d0 = 0; d0 < size[0]; ++d0) {
          int64_T d0BigVol = (offset[0] + d0);
          float* ptr_d0    = d0BigVol*stride[0] + ptr_d1; // ptr element volume
          bool   d0InRange = (d0BigVol >= 0) && (d0BigVol < sizeBigVol[0]);

          if (d3InRange && d2InRange && d1InRange && d0InRange)
            *pe_mat = *ptr_d0; // copy the single element if in range
          else
            *pe_mat = 0.0; // set to zero if out of range
          
          // advance to next matrix element
          pe_mat += the_mat.H;
        } // d0
      } // d1
    } // d2
  } // d3
  
}

void subvol4D::copy_and_inc_from_row (matw& the_mat, mwSize row)
{
  // TODO: code refactoring. almost the same code with copy_to_matw_row

  // init output: ptr element matrix
  float* pe_mat = the_mat.beg + row; 

  // For the sub volume, scan dim3,...,dim0. At each dim,
  // consider three cases: underflow, regular walking and overflow
  for (mwSize d3 = 0; d3 < size[3]; ++d3) {
    int64_T d3BigVol = (offset[3] + d3);
    float* ptr_d3    = d3BigVol*stride[3] + this->beg;
    bool   d3InRange = (d3BigVol >= 0) && (d3BigVol < sizeBigVol[3]);

    for (mwSize d2 = 0; d2 < size[2]; ++d2) {
      int64_T d2BigVol = (offset[2] + d2);
      float* ptr_d2    = d2BigVol*stride[2] + ptr_d3;
      bool   d2InRange = (d2BigVol >= 0) && (d2BigVol < sizeBigVol[2]); 

      for (mwSize d1 = 0; d1 < size[1]; ++d1) {
        int64_T d1BigVol = (offset[1] + d1);
        float* ptr_d1    = d1BigVol*stride[1] + ptr_d2;
        bool   d1InRange = (d1BigVol >= 0) && (d1BigVol < sizeBigVol[1]);

        for (mwSize d0 = 0; d0 < size[0]; ++d0) {
          int64_T d0BigVol = (offset[0] + d0);
          float* ptr_d0    = d0BigVol*stride[0] + ptr_d1; // ptr element volume
          bool   d0InRange = (d0BigVol >= 0) && (d0BigVol < sizeBigVol[0]);

          if (d3InRange && d2InRange && d1InRange && d0InRange)
            *ptr_d0 += *pe_mat; // copy and increment the single element if in range
          //else
          //  do nothing if out of range

          // advance to next matrix element
          pe_mat += the_mat.H;
        } // d0
      } // d1
    } // d2
  } // d3

}


} // namespace


//// impl of public methods
conv3d_blas_cpu::conv3d_blas_cpu()
{

}

conv3d_blas_cpu::conv3d_blas_cpu(const conv3d& obj)
{
  for (int i = 0; i < 6; ++i) pad[i]  = obj.pad[i];
  for (int i = 0; i < 3; ++i) stride[i] = obj.stride[i];

  X  = obj.X;
  dX = obj.dX;
  Y  = obj.Y;
  dY = obj.dY;
  F  = obj.F;
  dF = obj.dF;
  B  = obj.B;
  dB = obj.dB;

  ct = obj.ct;
}

void conv3d_blas_cpu::fprop()
{
  create_Y();
  init_convmat();
  init_u(); 

  // iterate over each training instance
  mwSize N = X.getSizeAtDim(4);
  for (mwSize i = 0; i < N; i++) {
    // make phiX: the convolution matrix
    vol_to_convmat(X, i);

    // convolution: Y_ = phiX * F_
    matw F_ = make_F_();
    matw Y_ = make_Y_(i);
    AxBtoC(convmat, F_, Y_, true); // overwrite Y_

    // plus the bias: Y_ += u * B
    matw B_ = make_B_();
    AxBtoC(u, B_, Y_, false); // accumulation on Y_
  }

  free_u();
  free_convmat();
}

void conv3d_blas_cpu::bprop()
{
  check_X_size();
  create_dX();
  create_dF();
  create_dB();
  init_convmat();
  init_u();

  mwSize N = X.getSizeAtDim(4);
  matw dF_ = make_dF_();
  matw dB_ = make_dB_();
  for (mwSize i = 0; i < N; ++i) {
    // make phiX: the convolution matrix
    vol_to_convmat(X, i);

    // dF += phiX' * dY_
    matw dY_ = make_dY_(i);
    ATxBtoC(convmat, dY_, dF_, false); // accumulation on dF_

    // dB += u' * dY
    ATxBtoC(u, dY_, dB_, false); // accumulation on dB_

    // dphiX = dY * F'
    matw F_ = make_F_();
    // safe to reuse convmat memory, remember to overwrite it!
    AxBTtoC(dY_, F_, convmat, true);
    // dX(:,:,:,:,i) <-- dphiX
    vol_from_convmat(dX, i);
  }

  free_u();
  free_convmat();
}

//// Impl of helper: fprop
matw conv3d_blas_cpu::make_F_()
{
  matw F_;
  F_.beg = (float*)F.getDataBeg();
  F_.H   = numelVol(F) * F.getSizeAtDim(3);
  F_.W   = F.getSizeAtDim(4);

  return F_;
}

matw conv3d_blas_cpu::make_Y_(mwSize i)
{
  matw Y_;
  Y_.beg = getVolInstDataBeg<float>(Y, i);
  Y_.H   = numelVol(Y);
  Y_.W   = Y.getSizeAtDim(3);

  return Y_;
}

matw conv3d_blas_cpu::make_B_()
{
  matw B_;
  B_.beg = (float*)B.getDataBeg();
  B_.H   = 1;
  B_.W   = numel(B);

  return B_;
}

//// Impl of helper: bprop
matw conv3d_blas_cpu::make_dY_(mwSize i)
{
  matw dY_;
  dY_.beg = getVolInstDataBeg<float>(dY, i);
  dY_.H   = numelVol(dY);
  dY_.W   = dY.getSizeAtDim(3);

  return dY_;
}

matw conv3d_blas_cpu::make_dF_()
{
  matw dF_;
  dF_.beg = (float*)dF.getDataBeg();
  dF_.H   = numelVol(dF) * dF.getSizeAtDim(3);
  dF_.W   = dF.getSizeAtDim(4);

  return dF_;
}

matw conv3d_blas_cpu::make_dB_()
{
  matw dB_;
  dB_.beg = (float*)dB.getDataBeg();
  dB_.H   = 1;
  dB_.W   = numel(dB);
  
  return dB_;
}

//// Impl of helper: the stacked matrix storing phiX or dphiX
void conv3d_blas_cpu::init_convmat()
{
  assert( (Y.pa_cpu != 0) || (dY.pa_cpu != 0) );
  if (Y.pa_cpu != 0) // in FPROP, Y has been set
    convmat.H = numelVol(Y);
  else // (dY != 0), in BPROP, dY has been set
    convmat.H = numelVol(dY);

  convmat.W = numelVol(F) * F.getSizeAtDim(3);
  mwSize nelem = convmat.H * convmat.W;
  convmat.beg = (float*)mxCalloc( nelem, sizeof(float) );
  // mxCalloc assures the initialization with all 0s ! 
}

void conv3d_blas_cpu::free_convmat()
{
  mxFree( (void*)convmat.beg );
}

void conv3d_blas_cpu::vol_to_convmat(xpuMxArrayTW &pvol, mwSize iInst)
{
  cpy_convmat_vol<VOL_TO_CONVMAT>(pvol, iInst);
}

void conv3d_blas_cpu::vol_from_convmat(xpuMxArrayTW &pvol, mwSize iInst)
{
  cpy_convmat_vol<VOL_FROM_CONVMAT>(pvol, iInst);
}

void conv3d_blas_cpu::init_u()
{
  assert( (Y.pa_cpu != 0) || (dY.pa_cpu != 0) );
  if (Y.pa_cpu != 0)
    u.H = numelVol(Y);
  else // (dY != 0)
    u.H = numelVol(dY);

  u.W = 1;
  mwSize nelem = u.H * u.W ;
  u.beg = (float*)mxMalloc( nelem * sizeof(float) );

  // make sure all one
  for (int i = 0; i < nelem; i++)
    u.beg[i] = 1.0;
}

void conv3d_blas_cpu::free_u()
{
  mxFree( (void*)u.beg );
}
