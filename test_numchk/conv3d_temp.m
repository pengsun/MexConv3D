%% data
szX = [3,2,2, 5,7];
szF = [3,2,2, 5,4];
szB = [1, szF(end)];
X = ones(szX, 'single');
F = ones(szF, 'single') / prod(szF(1:4));
B = zeros(szB, 'single');

iX = [2,1,1, 4,3];
ep = 1e-4;

%% num appro
deltaX = zeros(szX, 'single');
deltaX(iX(1), iX(2), iX(3), iX(4), iX(5)) = ep;
%
Y2 = mex_conv3d( X + deltaX, F, B);
z2 = sum( Y2(:) );
%
Y1 = mex_conv3d( X - deltaX, F, B);
z1 = sum( Y1(:) );
%
dzdep_app = (z2 - z1)/(2*ep);

%% by calculation
Y = mex_conv3d(X,F,B);
dzdy = ones(size(Y), 'single');
[dzdx,~,~] = mex_conv3d(X,F,B, dzdy);
dzdep = dzdx(iX(1), iX(2), iX(3), iX(4), iX(5));