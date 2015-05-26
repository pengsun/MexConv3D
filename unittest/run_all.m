function run_all ()
%% config
% rng(6345, 'twister');
type = 'cpu';
% type = 'gpu';
%% test Conv3d
if strcmp(type,'cpu')
  dg = @t_c3d.dg_cpu;
else
  dg = @t_c3d.dg_gpu;
end
run_all_conv3d(dg);
%% test MaxPool3d
if strcmp(type,'gpu')
  dg = @t_mp3d.dg_cpu;
else
  dg = @t_mp3d.dg_gpu;
end
run_all_mp3d(dg);


function run_all_conv3d (dg)
t_c3d.tc_1(dg);
t_c3d.tc_2(dg);
t_c3d.tc_3(dg);
t_c3d.tc_4(dg);
t_c3d.tc_5(dg);

function run_all_mp3d (dg)
t_mp3d.tc_1(dg);
t_mp3d.tc_2(dg);
t_mp3d.tc_3(dg);
t_mp3d.tc_4(dg);
t_mp3d.tc_5(dg);