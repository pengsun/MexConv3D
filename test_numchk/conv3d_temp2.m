%% data
szX = [3,2,2, 5,1];
szF = [3,2,2, 5,4];
szB = [1, szF(end)];
X = ones(szX, 'single');
F = ones(szF, 'single') / prod(szF(1:4));
B = zeros(szB, 'single');

ep = 1e-4;

%% num appro
deltaX = ep .* ones(szX, 'single');
%
Y2 = mex_conv3d( X + deltaX, F, B);
z2 = sum( Y2(:) );
%
Y1 = mex_conv3d( X - deltaX, F, B);
z1 = sum( Y1(:) );
%
dzdx_app = (z2 - z1) ./ (2 * deltaX);

%% by calculation
Y = mex_conv3d(X,F,B);
dzdy = ones(size(Y), 'single');
[dzdx,~,~] = mex_conv3d(X,F,B, dzdy);
%%
tmp = abs(dzdx_app - dzdx);