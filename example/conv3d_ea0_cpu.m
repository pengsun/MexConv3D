%% data
szX = [8,8,8, 5,9];
szF = [3,2,2, 5,4];
szB = [1, szF(end)];
X = rand(szX, 'single');
F = rand(szF, 'single');
B = rand(szB, 'single');
% X = getSeqMat(szX);
% F = getSeqMat(szF);
% B = getSeqMat(szB);
%% fprop
Y = mex_conv3d(X,F,B);
%% bprop
dZdY = rand(size(Y), 'single');
[dZdX,dZdF,dZdB] = mex_conv3d(X,F,B, dZdY);