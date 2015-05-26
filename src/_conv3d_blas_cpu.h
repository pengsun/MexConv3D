#pragma once
#include "conv3d.h"
#include "wrapperBlas.h"

//// conv3d: cpu version
struct conv3d_blas_cpu : public conv3d {
  conv3d_blas_cpu ();
  conv3d_blas_cpu (const conv3d& rhs);

  void fprop ();
  void bprop ();

private:
  // helper: fprop
  matw make_F_ ();
  matw make_Y_ (mwSize i);
  matw make_B_ ();

  // helper: bprop
  matw make_dY_ (mwSize i);
  matw make_dF_ ();
  matw make_dB_ ();

  // helper: the stacked matrix storing phiX or dphiX
  void init_convmat ();
  void free_convmat ();
  void vol_to_convmat   (xpuMxArrayTW &pvol, mwSize iInst); // im2row
  void vol_from_convmat (xpuMxArrayTW &pvol, mwSize iInst); // row2im
  matw convmat;

  void init_u ();
  void free_u ();
  matw u;

private:
  // helper for vol_to_convmat and convmat_to_vol
  enum DIR {VOL_TO_CONVMAT, VOL_FROM_CONVMAT};
  template<DIR how> void cpy_convmat_vol (xpuMxArrayTW &pvol, mwSize iInst) {
    // v: [H,   W,   D,   P]
    // F: [H',  W',  D',  P]
    // Y: [H'', W'', D'', 1]
    // convmat: [H''W''D''  H'W'D'P]

    // the big volume size and the sub volume
    mwSize H = pvol.getSizeAtDim(0), W = pvol.getSizeAtDim(1), 
      D = pvol.getSizeAtDim(2), P = pvol.getSizeAtDim(3);
    subvol4D sv;
    sv.beg = getVolInstDataBeg<float>(pvol, iInst);
    for (int i = 0; i < 4; ++i) sv.size[i] = F.getSizeAtDim(i);
    sv.sizeBigVol[0] = H;
    sv.sizeBigVol[1] = W;
    sv.sizeBigVol[2] = D;
    sv.sizeBigVol[3] = P;
    sv.stride[0] = 1; // always
    sv.stride[1] = H;
    sv.stride[2] = H*W;
    sv.stride[3] = H*W*D;

    // iterate over the big volume... 
    // ...and set the offset for the sub volume attaching to the big volume
    int64_T dim2_beg = -static_cast<int64_T>(pad[4]), 
            dim2_end =  static_cast<int64_T>(D + pad[5]);
    int64_T dim1_beg = -static_cast<int64_T>(pad[2]),
            dim1_end =  static_cast<int64_T>(W + pad[3]);
    int64_T dim0_beg = -static_cast<int64_T>(pad[0]), 
            dim0_end =  static_cast<int64_T>(H + pad[1]);
    int64_T FH = (int64_T)F.getSizeAtDim(0), 
            FW = (int64_T)F.getSizeAtDim(1), 
            FD = (int64_T)F.getSizeAtDim(2); 
    mwSize row = 0;

    sv.offset[3] = 0; // never slide at dim3 !
    for (int64_T k = dim2_beg; k < (dim2_end - FD + 1); k += this->stride[2]) { // slide at dim2
      sv.offset[2] = k;

      for (int64_T j = dim1_beg; j < (dim1_end - FW + 1); j += this->stride[1]) { // slide at dim1
        sv.offset[1] = j;

        for (int64_T i = dim0_beg; i < (dim0_end - FH + 1); i += this->stride[0]) { // slide at dim0
          sv.offset[0] = i;

          if (how == VOL_TO_CONVMAT)
            sv.copy_to_row(convmat, row);
          else // VOL_FROM_CONVMAT
            sv.copy_and_inc_from_row(convmat, row);

          // step to next row, should be consistent with i,j,k,p
          ++row;
        } // i
      } // j
    }// k 
  };

};

//// helper 