%% data
sz = [9,9,9, 3,7];
x = rand(sz, 'single');
% fprop
[y, ind] = mex_maxpool3d(x, 'pool',3);
% bprop (in ConvNet it should be dy, here we just use y for illustration)
xx = mex_maxpool3d(y,ind, 'pool',3);
%% validate
ix  = 4:6;
iy = 2;
yvalue = y(iy,iy,iy, 2,5);
subx = x(ix,ix,ix, 2,5);
subxx = xx(ix,ix,ix, 2,5);
%
assert( yvalue == max(subx(:)) )
assert( yvalue == max(subxx(:)) )
[~, im_x]  = max( subx(:) );
[~, im_xx] = max( subxx(:) );
assert( im_x == im_xx )

disp(yvalue)