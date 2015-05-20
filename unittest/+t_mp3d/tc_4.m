function tc_4()
sz     = [9, 3, 5, 7,6];
pool   = [2, 3, 4];
stride = [2, 1, 2];
pad    = [1,0,  0,0, 3,0];
desc = 'cpu array, regular case, pool > stride, non-zero pad';

h = t_mp3d.dg_cpu(sz, pool, stride, pad, desc);

hu = t_mp3d.unit(h);
hu.run();