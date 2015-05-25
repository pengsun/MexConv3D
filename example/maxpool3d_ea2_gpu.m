%% data
sz = [7,8,5, 5,9];
pool   = [3,2,4];
stride = [2,1,2];
pad    = [1,1,  0,0, 2,1];
x = gpuArray.rand(sz, 'single');
%% fprop
[y, ind] = mex_maxpool3d(x,...
  'pool',pool, 'stride',stride, 'pad',pad);
%% bprop (in ConvNet it should be dy, here we just use y for illustration)
xx = mex_maxpool3d(y,ind,...
  'pool',pool, 'stride',stride, 'pad',pad);