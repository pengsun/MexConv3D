function conv3d_vs_matconvnet_bprop_cpu()
disp( 'dim1, dim2 as 2d conv' )
szX = [13,10,1, 2, 5];
szF = [3, 5, 1, 2, 4];
szB = [1,4];
pad = [0,2, 1,5, 0,0];
stride = [2, 3, 1];

cmp_3d2d(szX, szF, szB,...
  pad, stride,...
  pad(1:4), stride(1:2) )


disp('dim1, dim3 as 2d conv')
szX = [13, 1, 10, 2, 5];
szF = [3,  1, 5,  2, 4];
szB = [1, 4];
pad = [0,2, 0,0, 1,5];
stride = [2, 1, 3];

cmp_3d2d(szX, szF, szB,...
  pad, stride,...
  [pad(1:2),pad(5:6)], [stride(1),stride(3)] );


disp('dim2, dim3 as 2d conv')
szX = [1, 13,  10, 2, 5];
szF = [1, 3,   5,  2, 4];
szB = [1, 4];
pad = [0,0, 0,2, 1,5];
stride = [1, 2, 3];

cmp_3d2d(szX, szF, szB,...
  pad, stride,...
  pad(3:6), stride(2:3) );


function cmp_3d2d(szX, szF, szB, pad3d, stride3d, pad2d, stride2d)

ran = 50;

% gen data
X = ran * randn(szX, 'single');
F = ran * randn(szF, 'single');
B = ran * randn(szB, 'single');
Y = mex_conv3d(X,F,B,...
  'pad', pad3d,...
  'stride', stride3d);
szY = size(Y);
dzdy = ran * randn(szY, 'single');

% 3d conv
[dX1,dF1,dB1] = mex_conv3d(X,F,B, dzdy,...
  'pad', pad3d,...
  'stride', stride3d);

% 2d conv: matconvnet
[dX2,dF2,dB2] = vl_nnconv(squeeze(X), squeeze(F), B, squeeze(dzdy),...
  'pad', pad2d,...
  'stride', stride2d );

% the size should be same (except for the singular dim)
disp('size(dX1) as 3d conv:')
disp(size(dX1))
disp('size(dX2) as 2d conv:')
disp(size(dX2))

disp('size(dF1) as 3d conv:')
disp(size(dF1))
disp('size(dF2) as 2d conv:')
disp(size(dF2))

disp('size(dB1) as 3d conv:')
disp(size(dB1))
disp('size(dB2) as 2d conv:')
disp(size(dB2))

% the values should be same
tau = ran * 1e-3; 

diff_dX = abs(dX1(:)-dX2(:));
fprintf('assert dX1 - dX2 is very small\n');
assert( all( diff_dX < tau ) );

diff_dF = abs(dF1(:)-dF2(:));
fprintf('assert dF1 - dF2 is very small\n');
assert( all( diff_dF < tau ) );

diff_dB = abs(dB1(:)-dB2(:));
fprintf('assert dB1 - dB2 is very small\n\n\n');
assert( all( diff_dB < tau ) );