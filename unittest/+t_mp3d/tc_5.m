function tc_5()
sz     = [30, 30, 30, 5, 16];
pool   = [4, 4, 4];
stride = [2, 2, 2];
pad    = [1,1,  1,1, 1,1];
desc = 'cpu array, regular case, pool = 2*stride, non-zero pad';

h = t_mp3d.dg_cpu(sz, pool, stride, pad, desc);

hu = t_mp3d.unit(h);
hu.run();