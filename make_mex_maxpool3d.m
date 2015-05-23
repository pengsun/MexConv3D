%%
dir_root = fileparts( mfilename('fullpath') );
dir_inc = fullfile(dir_root, 'src');
%% files to be compiled and linked
src = {};
src{end+1} = 'mex_maxpool3d.cpp';
src{end+1} = 'src/maxpool3d.cpp';
src{end+1} = 'src/_maxpool3d_cpu.cpp';
src = cellfun(@(f)(fullfile(dir_root,f)), src, 'UniformOutput',false);
%% options
opt = {};
opt{end+1} = '-g';

opt{end+1} = ['-I',dir_inc];
opt{end+1} = '-largeArrayDims';
% opt{end+1} = '-DVB';
opt{end+1} = '-outdir';
opt{end+1} = dir_root;
%% optoins: gpu 
opt{end+1} = '-DWITH_GPUARRAY';

% header: "gpu/mxGPUArray.h"
dir_gpu = fullfile('toolbox','distcomp','gpu','extern','include');
dir_gpu = fullfile(matlabroot, dir_gpu);
opt{end+1} = ['-I', dir_gpu]; 
% lib: gpu.lib for mxGPU* stuff
opt{end+1} = '-lgpu';
%% options: openmp
str = computer('arch');
switch str(1:3)
  case 'win' 
    opt{end+1} = 'COMPFLAGS=/openmp $COMPFLAGS';
    opt{end+1} = 'LINKFLAGS=/openmp $LINKFLAGS';
  otherwise
    opt{end+1} = 'CXXFLAGS="\$CXXFLAGS -fopenmp"';
    opt{end+1} = 'LDFLAGS="\$LDFLAGS -fopenmp"';
end
%% do it
mex(src{:}, opt{:});