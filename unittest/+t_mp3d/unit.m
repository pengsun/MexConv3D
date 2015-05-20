classdef unit
  %UNIT unit test for maxpool3d
  %   Detailed explanation goes here
  
  properties
    hdg; % handle to data generator
  end
  
  methods
    
    function ob = unit (hdg_)
      ob.hdg = hdg_;
    end
    
    function run (ob)
      print(ob);
        
      fprintf('verifying...\n');
      test_dX(ob);
      fprintf('done.\n\n');
    end
    
    function print(ob)
      fprintf('description: %s\n', ob.hdg.desc);
      
      szX = size5d( ob.hdg.X );
      fprintf('X: [%d %d %d %d %d]\n', szX);
      
      if (~isempty(ob.hdg.pool))
        fprintf('%s\n', fmt_opt('pool', ob.hdg.pool) );
      end
      
      if (~isempty(ob.hdg.stride))
        fprintf('%s\n', fmt_opt('stride', ob.hdg.stride) );
      end
      
      if (~isempty(ob.hdg.pad) )
        fprintf('%s\n', fmt_opt('pad', ob.hdg.pad) );
      end
      
    end
    
    function test_dX (ob)
      % for convenient typing
      [X, pool, stride, pad] = get_all(ob);
      
      % 
     	fprintf('fprop: generating Y and max index...\n');
      [Y, ind] = mex_maxpool3d(X,...
        'pool',pool, 'stride',stride, 'pad',pad);
      fprintf('Y: [%d %d %d %d %d]\n', size5d(Y) );
      % 
      fprintf('assert X(ind(:)) == Y\n');
      tmp = X(ind(:));
      assert( all( tmp(:)==Y(:) ) );
      %
      fprintf('generating all one dzdy...\n');
      dzdy = ones(size(Y), 'single'); % TODO: like
      %
      fprintf('bprop: generating dzdx...\n');
      dzdx = mex_maxpool3d(dzdy,ind,...
        'pool',pool, 'stride',stride, 'pad',pad);
      %
      fprintf('assert find(dzdx>0) == sort_ascend_unqique_max_index...\n');
      a1 = find( dzdx > 0 );
      a2 = sort( unique(ind(:)), 'ascend');
      assert( all( a1(:) == a2(:) ) );
    end

  end % methods
end % unit  

function [X, pool, stride, pad, ran] = get_all (ob)
  X = ob.hdg.X;
  pool   = ob.hdg.pool;
  stride = ob.hdg.stride;
  pad    = ob.hdg.pad;
  
  ran = ob.hdg.ran;
end % test_dX

