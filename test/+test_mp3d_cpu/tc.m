function tc(sz)
%TC1 Summary of this function goes here
%   Detailed explanation goes here


x = rand(sz, 'single');
[y, ind] = mex_maxpool3d(x);

xx = mex_maxpool3d(y,ind);


end

