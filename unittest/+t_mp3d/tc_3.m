function tc_3()
sz = [7,8,5, 5,9];
pool   = [3,2,4];
stride = [2,1,2];
pad    = [1,1,  0,0, 2,1];
desc = 'cpu array, regular case, pool > stride, non-zero pad';

h = t_mp3d.dg_cpu(sz, pool, stride, pad, desc);

hu = t_mp3d.unit(h);
hu.run();