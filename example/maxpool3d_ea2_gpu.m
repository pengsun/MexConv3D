%% data
sz = [7,8,5, 5,9]; % size for input: 3D volume + #feature maps + #instances
pool   = [3,2,4];          % 3D pooling window size
stride = [2,1,2];          % 3D stride 
pad    = [1,1,  0,0, 2,1]; % 3D lower/higher padding
x = gpuArray.rand(sz, 'single'); % Input data/feature maps
%% fprop
[y, ind] = mex_maxpool3d(x,...
  'pool',pool, 'stride',stride, 'pad',pad);
%% bprop
dzdy = rand(size(y),'like',y);
xx = mex_maxpool3d(dzdy,ind,size(x),...
  'pool',pool, 'stride',stride, 'pad',pad);