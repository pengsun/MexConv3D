#pragma once
#include "conv3d.h"
#include "mat_op.h"

//// conv3d: cpu version
struct conv3d_cpu : public conv3d {
  conv3d_cpu ();

  void fprop ();
  void bprop ();
  CALL_TYPE parse_and_set (int no, mxArray *vo[], int ni, mxArray const *vi[]);

private:
  void set_stride (mxArray const *pa);
  void set_pad    (mxArray const *pa);

  // helper: fprop
  void create_Y  ();
  matw make_F_ ();
  matw make_Y_ (mwSize i);
  matw make_B_ ();

  // helper: bprop
  void create_dX ();
  void create_dF ();
  void create_dB ();
  matw make_dX_ (mwSize i);
  matw make_dY_ (mwSize i);
  matw make_dF_ ();
  matw make_dB_ ();

  // helper: the stacked matrix storing phiX or dphiX
  void init_convmat ();
  void free_convmat ();
  void vol_to_convmat (const mxArray *pvol, mwSize iInst); // im2row
  void convmat_to_vol (mxArray *pvol, mwSize i);           // row2im
  matw convmat;

  void init_u ();
  void free_u ();
  matw u;
};

//// helper 