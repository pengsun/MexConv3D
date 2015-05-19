function tc_2()
szX = [8, 8, 8, 5, 9];
szF = [3, 2, 2, 5, 4];
szB = [1, 4];
stride = [1,1,1];
pad = [4,2, 3,1, 2,5];
desc = 'cpu array, regular case, non-zero pad';

h = t_c3d.dg_cpu(szX,szF,szB, stride, pad, desc);

hu = t_c3d.unit(h);
hu.run();


