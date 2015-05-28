#pragma once
#include "conv3d.h"
#include "wrapperBlas.h"

//// conv3d: gpu version TODO: impl idiom
struct conv3d_blas_gpu : public conv3d {
  conv3d_blas_gpu ();
  conv3d_blas_gpu (const conv3d& rhs);

  void fprop ();
  void bprop ();

  // helper types for implementation
  struct vol4d {
    float* beg;
    mwSize sz[4];
  };

  struct CpyVolConvmatImpl {
    // Source, Target
    vol4d vol_i; // X(:,:,:,:,i) or dX(:,:,:,:,i)
    matw  convmat;
    // other information
    mwSize szY[3];
    mwSize szF[3];
    mwSize stride[3];
    mwSize pad[6];
  };

private:
  // helper: fprop
  matw make_F_ ();
  matw make_Y_ (mwSize i);
  matw make_B_ ();

  // helper: bprop
  matw make_dY_ (mwSize i);
  matw make_dF_ ();
  matw make_dB_ ();

private: // helper: the stacked matrix storing phiX or dphiX
  CpyVolConvmatImpl make_initial_CpyVolConvmatImpl (const xpuMxArrayTW &vol);

  void init_convmat ();
  void free_convmat ();
  void vol_to_convmat   (CpyVolConvmatImpl &ip, xpuMxArrayTW &vol, mwSize iInst); // im2row
  void vol_from_convmat (CpyVolConvmatImpl &ip, xpuMxArrayTW &vol, mwSize iInst); // row2im
  matw convmat;

private: // helper for unit vector u
  void init_u ();
  void free_u ();
  matw u;
};

//// helper 