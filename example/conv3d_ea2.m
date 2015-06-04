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
dZdY = rand(size(Y), 'single');
[dZdX,dZdF,dZdB] = mex_conv3d(X,F,B, dZdY, 'pad',pad, 'stride',stride);