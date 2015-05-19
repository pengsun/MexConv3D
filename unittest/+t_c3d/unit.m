classdef unit
  %UNIT unit test for conv3d
  %   Detailed explanation goes here
  
  properties
    hdg; % handle to data generator
    ep;  % epsilon
    ran; % range
  end
  
  methods
    
    function ob = unit (hdg_)
      ob.hdg = hdg_;
      
      ob.ep = 1e-2;
      ob.ran = hdg_.ran;
    end
    
    function run (ob)
      print(ob);
      
      fprintf('verifying fprop: Y...\n');
      test_Y(ob);
      
      fprintf('verifying bprop: dX...\n');
      r = test_dX(ob.hdg.X, ob.hdg.F, ob.hdg.B,...
        ob.hdg.stride, ob.hdg.pad,...
        ob.ran, ob.ep);
      fprintf('avg relative diff: %7.6f %%\n', 100*mean(r) );
      
      test_dF(ob);
      test_dB(ob);
      
      fprintf('done.\n\n');
    end
    
    function print(ob)
      fprintf('description: %s\n', ob.hdg.desc);
      
      szX = size5d( ob.hdg.X );
      fprintf('X: [%d %d %d %d %d]\n', szX);
      
      szF = size5d( ob.hdg.F );
      fprintf('F: [%d %d %d %d %d]\n', szF);
      
      szB = [1, size(ob.hdg.B,2)];
      fprintf('B: [%d %d]\n', szB);
      
      if (~isempty(ob.hdg.stride))
        fprintf('%s\n', fmt_opt('stride', ob.hdg.stride) );
      end
      
      if (~isempty(ob.hdg.pad) )
        fprintf('%s\n', fmt_opt('pad', ob.hdg.pad) );
      end
      
    end
    
    function test_Y (ob)
      
    end
    
%     function test_dX (ob)
%      
%     end
    
    function test_dF (ob)
    end
    
    function test_dB (ob)
    end

  
  end % methods
end % unit  

function r = test_dX(X, F, B, stride, pad, ran, ep)

ind = randperm( numel(X) );
ind = ind(1:10);
for i = 1 : numel(ind)
  ii = ind(i);
  % by numeric loss difference
  szX = size(X);
  deltaX = zeros(szX, 'single'); % TODO: like 
  deltaX(ii) = ep .* ran;
  %
  Y2 = mex_conv3d( X + deltaX, F, B, 'stride',stride, 'pad',pad);
  z2 = sum( Y2(:) );
  %
  Y1 = mex_conv3d( X - deltaX, F, B,  'stride',stride, 'pad',pad);
  z1 = sum( Y1(:) );
  %
  dzdep_app = (z2 - z1)/(2*ep*ran);
  
  % by calculation
  Y = mex_conv3d(X,F,B, 'stride',stride, 'pad',pad);
  dzdy = ones(size(Y), 'single');
  [dzdx,~,~] = mex_conv3d(X,F,B, dzdy,  'stride',stride, 'pad',pad);
  dzdep = dzdx(ii);
  
  % compare
  r(i) = abs(dzdep_app - dzdep)/abs(dzdep+eps); %#ok<AGROW>
end

end % test_dX

