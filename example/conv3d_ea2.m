%% data
szX = [9,8,5, 5,9];
szF = [3,3,3, 5,4];
szB = [1, szF(end)];
X = rand(szX, 'single');
F = rand(szF, 'single');
B = rand(szB, 'single');

pad    = [1,2, 2,1, 3,4];
stride = [2,3,5];
%% fprop
Y = mex_conv3d(X,F,B, 'pad', pad, 'stride',stride);
%% bprop
dZdY = rand(size(Y), 'single');
[dZdX,dZdF,dZdB] = mex_conv3d(X,F,B, dZdY, 'pad',pad, 'stride',stride);