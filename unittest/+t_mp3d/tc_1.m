function tc_1()
sz = [8,8,8, 5,9];
pool   = [2, 2, 2];
stride = [2, 2, 2];
pad = [0 0 0 0 0 0];

% sz = [4,2,2, 1,1];
% pool   = [2, 2, 2];
% stride = [2, 2, 2];
% pad = [0 0 0 0 0 0];

desc = 'cpu array, regular case, pool = stride, no pad';

h = t_mp3d.dg_cpu(sz, pool, stride, pad, desc);

hu = t_mp3d.unit(h);
hu.run();