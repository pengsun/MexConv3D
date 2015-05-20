function tc_tmp()
sz = [4,4,1, 1,1];
pool   = [2, 2, 1];
stride = pool;
pad = [0 0 0 0 0 0];


desc = 'cpu array, regular case, pool = stride, no pad';

h = t_mp3d.dg_cpu(sz, pool, stride, pad, desc);

hu = t_mp3d.unit(h);
hu.run();