#include "conv3d.h"
#include "_conv3d_cpu.h"

//// Impl of conv3d
const char* conv3d::THE_CMD = 
  "Bad input or output arguments. The right way to call:\n"
  "Y = MEX_CONV3D(X,F,B); forward pass\n"
  "[dX,dF,dB] = MEX_CONV3D(X,F,B, dY); backward pass\n"
  "MEX_CONV3D(..., 'stride',s, 'pad',pad); options\n"
  "All arguments must be single.\n";

conv3d::conv3d()
{
  stride[0] = stride[1] = stride[2] = 1;
  pad[0] = pad[1] = pad[2] = pad[3] = pad[4] = pad[5] = 0;

}


//// impl of conv3d_ex
conv3d_ex::conv3d_ex(const char* msg)
  : exception(msg)
{

}


//// Impl of factory_c3d_homebrew
conv3d* factory_c3d_homebrew::parse_and_create(int no, mxArray *vo[], int ni, mxArray const *vi[])
{

  // fprop or bprop?
  conv3d holder;
  int n_opt = -1;
  if (no == 1) {
    if ( ni < 3)
      throw conv3d_ex("Too few input arguments for fprop(). At least three: X, F, B.");

    holder.X.setMxArray( (mxArray*) vi[0] );
    holder.F.setMxArray( (mxArray*) vi[1] );
    holder.B.setMxArray( (mxArray*) vi[2] );
    if ( holder.X.getElemType() != mxSINGLE_CLASS || 
         holder.F.getElemType() != mxSINGLE_CLASS ||
         holder.B.getElemType() != mxSINGLE_CLASS) 
      throw conv3d_ex("The first three arguments X, F, B should be all SINGLE type.");

    holder.ct = conv3d::FPROP;
    n_opt = 3;
  } 
  else if (no == 3) {
    if ( ni < 4 ) 
      throw conv3d_ex("Too few input arguments for bprop(). At least four: X, F, B, dZdY.");

    holder.X.setMxArray( (mxArray*) vi[0] );
    holder.F.setMxArray( (mxArray*) vi[1] );
    holder.B.setMxArray( (mxArray*) vi[2] );
    holder.dY.setMxArray( (mxArray*) vi[3] );
    if (holder.X.getElemType() != mxSINGLE_CLASS || 
        holder.F.getElemType() != mxSINGLE_CLASS ||
        holder.B.getElemType() != mxSINGLE_CLASS ||
        holder.dY.getElemType() != mxSINGLE_CLASS)
      throw conv3d_ex("The first four arguments X, F, B, dZdY should be SINGLE type");

    holder.ct = conv3d::BPROP;
    n_opt = 4;
  } 
  else {
    throw conv3d_ex("Unrecognized way of calling."
      "The output should be either Y (fprop) or [dX,dF,dB] (bprop). \n");
  }

  set_options(holder, n_opt, ni, vi);

  // TODO: gpu version here
  return new conv3d_cpu(holder);
}

void factory_c3d_homebrew::set_options(conv3d &holder, int opt_beg, int ni, mxArray const *vi[])
{
  // parse option/value pairs
  if ( ((ni-opt_beg)%2) != 0 )
    throw conv3d_ex("Imbalanced option/value pairs.");
  for (int i = opt_beg; i < ni; i+=2) {
    if (isStrEqual(vi[i], "stride"))   set_stride(holder, vi[i+1]);
    else if (isStrEqual(vi[i], "pad")) set_pad(holder, vi[i+1]);
    else                               throw conv3d_ex("Unrecognized option/value pairs.");
  } // for i
}

void factory_c3d_homebrew::set_stride(conv3d &holder, mxArray const *pa )
{
  if ( !setCArray<mwSize, 3>(pa, holder.stride) )
    throw conv3d_ex("The length of option stride should be either 1 or 3.");
}

void factory_c3d_homebrew::set_pad(conv3d &holder, mxArray const *pa )
{
  if ( !setCArray<mwSize, 6>(pa, holder.pad) )
    throw conv3d_ex("The length of option pad should be either 1 or 6.");
}

