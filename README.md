# MexConv3D
Matlab mex implementation (with both CPU and GPU version) of the basic operations for 3D (volume) ConvNet. Potential use: video, volume data (CT, MR), etc. The APIs and conventions are consistent with [matconvnet](https://github.com/vlfeat/matconvnet). 

## Overview
`mex_conv3d.m` is provided for 3D convoluation, while `mex_maxpool3d.m` for 3D max pooling. Their calling conventions are consistent with `vl_nnconv.m` and `vl_nnpool.m` in [matconvnet](https://github.com/vlfeat/matconvnet), respectively.

In this project only the most basic building blocks are provided. For a ready-to-use 3D ConvNet in Matlab, one needs a high-level wrapper like the `vl_simplenn.m` in [matconvnet](https://github.com/vlfeat/matconvnet) or [MatConvDAG](https://github.com/pengsun/MatConvDAG)

## Install
Step by step:

* Run `make_all.m` in root directory to compile the mex files 
  * CUDA toolkit needed if enabling GPU
* Run `setup_path.m` in root directory to add path
* (Optional) CD to folder `unittest` and run `run_all.m` to verify everything works well

MexConv3D has been tested in the following environment:

* Matlab R2014a + Windows 8.1 + Visual Studio 2012
* Matlab R2014a + Ubuntu 12.04 + GCC 4.8.2

## Usage
For 3D Convolution:
```Matlab
%% data
szX = [9,8,5, 5,9];  % input size: 3D volume + #feature maps + #instances
szF = [3,3,3, 5,4];  % filter size: 3D volume + #input feature maps + #output feature maps
szB = [1, szF(end)]; % bias size: #output feature maps
X = gpuArray.rand(szX, 'single');
F = gpuArray.rand(szF, 'single');
B = gpuArray.rand(szB, 'single');

pad    = [1,2, 2,1, 3,4]; % 3D higher/lower padding
stride = [2,3,5];         % 3D stride
%% fprop
Y = mex_conv3d(X,F,B, 'pad', pad, 'stride',stride);
%% bprop
dZdY = rand(size(Y), 'like', Y);
[dZdX,dZdF,dZdB] = mex_conv3d(X,F,B, dZdY, 'pad',pad, 'stride',stride);
```

For 3D Max Pooling: 
```Matlab
%% data
sz = [7,8,5, 5,9]; % size for 3D volume + #feature maps + #instances
pool   = [3,2,4];          % 3D pooling window size
stride = [2,1,2];          % 3D stride 
pad    = [1,1,  0,0, 2,1]; % 3D higher/lower padding
x = gpuArray.rand(sz, 'single'); % Input data/feature maps
%% fprop
[y, ind] = mex_maxpool3d(x,...
  'pool',pool, 'stride',stride, 'pad',pad);
%% bprop
dzdy = rand(size(y),'like',y);
xx = mex_maxpool3d(dzdy,ind,size(x),...
  'pool',pool, 'stride',stride, 'pad',pad);
```

See more scripts in directory `example`. Type `help mex_conv3d` or `help mex_maxpool3d` for doc. See `README.md` in each folder (if any) to understand the folder layout and the purpose.
