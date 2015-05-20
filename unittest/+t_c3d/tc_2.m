function tc_2()
%%
szX = [8, 8, 8, 5, 3];
szF = [3, 2, 2, 5, 4];
szB = [1, 4];
stride = [1,1,1];
pad = [1,2, 3,1, 2,3];
desc = 'cpu array, regular case, non-zero pad';

h = t_c3d.dg_cpu(szX,szF,szB, stride, pad, desc);

hu = t_c3d.unit(h);
hu.run();
%%
szX = [7, 8, 6, 5, 4];
szF = [3, 2, 2, 5, 3];
szB = [1, 3];
stride = [1,1,1];
pad = [2,1, 0,1, 3,0];
desc = 'cpu array, regular case, non-zero pad';

h = t_c3d.dg_cpu(szX,szF,szB, stride, pad, desc);

hu = t_c3d.unit(h);
hu.run();