%MEX_POOL3D Volume max pooling for 3D convnet
%  Y = mex_pool3d(X, pool); forward pass
%  [dZdX] = mex_pool3d(X, pool, dZdY); backward pass
%
%  Input:
%   X: [H,W,D,M,N]. Input volume or feature maps. H, W, D are volume's 
%   height, width and depth, respectively. M is #volumes (or #feature maps).
%   N is #instances.
%   pool: [PH,PW,PD] or [P]. Pooling 3D window size. PH, PW, PD are the 
%   height, width and depth, respectively. P is the size for all. Default 
%   to 2.
%
%  Output:
%   Y: [Ho,Wo,Do,M,N]. Output feature maps
%   dZdX: [H,W,D,M,N]. Delta w.r.t X where Z means loss.  
%
