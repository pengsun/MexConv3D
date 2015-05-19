dir_name = 't_c3d';
this_dir = fileparts( mfilename('fullpath') );
tar_dir  = fullfile(this_dir, ['+',dir_name]);

fns = dir(tar_dir);
for i = 1 : numel(fns)
  if ( fns(i).isdir ), continue; end
  
  % expect a test case with name 'tc_*'
  [~,nm,ext] = fileparts( fns(i).name );
  if ( strcmp(nm(1:3),'tc_') && strcmp(ext,'.m') )
    cmd = sprintf('%s.%s()', dir_name, nm);
    fprintf('running %s...\n', cmd);
    
    try
      eval( cmd );
    catch er
      fprintf('error occured!\n');
      disp(er.message);
      continue;
    end % try
    
  end % if
  
end