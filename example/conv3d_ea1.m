%% data
szX = [8,8,8, 5,9];
szF = [3,2,2, 5,7];
szB = [1, szF(end)];
X = rand(szX, 'single');
F = rand(szF, 'single');
B = rand(szB, 'single');
%% fprop
Y = mex_conv3d(X,F,B);
%% bprop
dZdY = rand(size(Y), 'single');
[dZdX,dZdF,dZdB] = mex_conv3d(X,F,B, dZdY);
%% validate size
% size Y
szY123 = szX(1:3)- szF(1:3) + 1;
szY = [szY123, szF(5), szX(5)];
assert( all( szY == size(Y) ) );

% size dX, dF, dB
assert( all( size(X)==size(dZdX) ) );
assert( all( size(F)==size(dZdF) ) );
assert( all( size(B)==size(dZdB) ) );
%% validate results