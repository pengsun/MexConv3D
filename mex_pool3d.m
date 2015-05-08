%MEX_POOL3D Volume max pooling for 3D convnet
%  [Y,ind] = MEX_POOL3D(X); forward pass
%  dZdX = MEX_POOL3D(dZdY, ind); backward pass
%  MEX_POOL3D(..., 'pool',pool, 'stride',s, 'pad',pad); options
%
%  Input:
%   X: [H,W,D,M,N]. Volume at input port or feature maps. H, W, D are volume's 
%   height, width and depth, respectively. M is #volumes (or #feature maps).
%   N is #instances.
%   dZdY: [Ho,Wo,Do,M,N]. Delta signal at output port. Z means loss.
%   ind: see ind Output
%
% Options:  
%   pool: [PH,PW,PD] or [P]. Pooling 3D window size. PH, PW, PD are the 
%   height, width and depth, respectively. P is the size for all. Default 
%   to P = 2.
%   s: [sH,sW,SD] or [s]. Default to s = 2
%   pad: [PH,PW,PD] or [P]. Padding size
%
%  Output:
%   Y: [Ho,Wo,Do,M,N]. Feature maps at output port
%   ind: [Ho,Wo,Do,M,N]. Linear index of the max elements to X: Y = X(ind)
%   dZdX: [H,W,D,M,N]. Delta signal at input port. Z means loss.  
%