clear all
load('FaceData.mat')
sz = 25;

figure()
[n,~] = size(data3);
c = linspace(1,10,n);
Ylle = lle(data3'+1e-5*rand(size(data3')),50,3);
% scatter(Ylle(1,:),Ylle(2,:),sz,c,'filled')
scatter3(Ylle(1,:),Ylle(2,:),Ylle(3,:),sz,c,'filled')