function tc_1()
%%
szX = [9, 9, 1, 1, 1];
szF = [3, 3, 1, 1, 1];
szB = [1, 1];
stride = [1,1,1];
pad = [0,0, 0,0, 0,0];
desc = 'cpu array, regular case, no padding, stride 1';

h = t_c3d.dg_cpu(szX,szF,szB, stride, pad, desc);

hu = t_c3d.unit(h);
hu.run();
%%
szX = [8, 8, 8, 2, 1];
szF = [3, 3, 3, 2, 4];
szB = [1, 4];
stride = [1,1,1];
pad = [0,0, 0,0, 0,0];
desc = 'cpu array, regular case, no padding, stride 1';

h = t_c3d.dg_cpu(szX,szF,szB, stride, pad, desc);

hu = t_c3d.unit(h);
hu.run();
