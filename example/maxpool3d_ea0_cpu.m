%% data
sz = [8,8,8, 5,9];
x = rand(sz, 'single');
%% fprop
[y, ind] = mex_maxpool3d(x);
%% bprop 
dzdy = ones(size(y), 'single');
dzdx = mex_maxpool3d(dzdy,ind);
%%
a1 = x( dzdx > 0 );
a2 = x( sort(ind(:),'ascend') );
assert( all( a1(:) == a2(:) ) );