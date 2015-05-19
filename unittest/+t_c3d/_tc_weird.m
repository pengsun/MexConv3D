function tc_weird()
szX = [8, 8, 8, 5, 9];
szF = [10, 2, 2, 5, 4];
szB = [1, 4];
stride = [];
pad = [];
desc = 'cpu array, F size > X size, should raise an error';

h = t_c3d.dg_cpu(szX,szF,szB, stride, pad, desc);

hu = t_c3d.unit(h);
hu.run();


