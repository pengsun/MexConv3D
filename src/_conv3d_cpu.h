#pragma once
#include "conv3d.h"

struct conv3d_cpu : public conv3d {
  conv3d_cpu ();

  void fprop ();
  void bprop ();
  CALL_TYPE parse_and_set (int no, mxArray *vo[], int ni, mxArray const *vi[]);

private:
  void set_stride (mxArray const *pa);
  void set_pad    (mxArray const *pa);

  void create_Y   ();
  void create_dX  ();
};