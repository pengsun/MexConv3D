%%
dir_root = fileparts( mfilename('fullpath') );
dir_inc = fullfile(dir_root, 'src');
%% files
src = {};
src{end+1} = 'mex_conv3d.cpp';
src{end+1} = 'src/conv3d.cpp';
src{end+1} = 'src/_conv3d_cpu.cpp';
src{end+1} = 'src/mat_op.cpp';
src = cellfun(@(f)(fullfile(dir_root,f)), src, 'UniformOutput',false);
%% options
opt = {};
opt{end+1} = '-g';

opt{end+1} = ['-I',dir_inc];
opt{end+1} = '-largeArrayDims';
% opt{end+1} = '-DVB';
opt{end+1} = '-outdir';
opt{end+1} = dir_root;
%% options: blas
% str = computer('arch');
% switch str(1:3)
%   case 'win' 
% %     blaslib = fullfile(matlabroot,'extern','lib',computer('arch'),'microsoft',...
% %       'libmwblas.lib');
% %     opt{end+1} = blaslib;
%     opt{end+1} = '-lmwblas';
%   otherwise
%     opt{end+1} = '-lmwblas';
% end

opt{end+1} = '-lmwblas'; % works for both Windows or Unix?
%% options: openmp
% str = computer('arch');
% switch str(1:3)
%   case 'win' 
%     opt{end+1} = 'COMPFLAGS=/openmp $COMPFLAGS';
%     opt{end+1} = 'LINKFLAGS=/openmp $LINKFLAGS';
%   otherwise
%     opt{end+1} = 'CXXFLAGS="\$CXXFLAGS -fopenmp"';
%     opt{end+1} = 'LDFLAGS="\$LDFLAGS -fopenmp"';
% end
%% do it
mex(src{:}, opt{:});