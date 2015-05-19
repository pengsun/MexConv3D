classdef dg_cpu
  %DG_CPU Summary of this class goes here
  %   Detailed explanation goes here
  
  properties
    X;
    F;
    B;
    stride;
    pad;
    
    desc;
    ran;
  end
  
  methods
    function ob = dg_cpu(szX, szF, szB, stride, pad, desc)
      
      ob.ran = 100;
      
      ob.X = ob.ran * randn(szX, 'single');
      ob.F = ob.ran * randn(szF, 'single');
      ob.B = ob.ran * randn(szB, 'single');
      
      ob.stride = [];
      if (~isempty(stride))
        ob.stride = stride;
      end
      
      ob.pad = [];
      if (~isempty(pad))
        ob.pad = pad;
      end
      
      ob.desc = desc;
    end % dg_cpu
    
  end % methods
  
end

