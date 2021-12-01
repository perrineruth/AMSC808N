%% problem 1 of Proj 3 AMSC 808N
% S curve no noise
close all
clear


% load data matrix (it's called data3)
load("ScurveData.mat");
[n,~] = size(data3);
expDim = 2;
% for plotting in color (emphasize the change over the s shape to see if it
% folds back)
c = linspace(1,5,n)+repmat(linspace(1,5,32),1,11);
sz = 25;


% actually run each dimension reduction algorithm
% 1 PCA
% first center the data
Ycentered = data3 - ones(n,1)*sum(data3,1)/n;
[U,Sigma,~] = svd(data3','econ');
Ypca = Ycentered*U(:,1:expDim); % features
scatter(Ypca(:,1),Ypca(:,2),sz,c,'filled')

% 2 Isomap
figure()
Yiso = isomap(data3,5,2);
scatter(Yiso(:,1),Yiso(:,2),sz,c,'filled')

% 3 LLE
figure()
k = 7;                      % # nearest neighbors
Ylle = lle(data3',k,expDim);
scatter(Ylle(1,:),Ylle(2,:),sz,c,'filled')

% 4 t-SNE
figure()
perp = 10;
[Yt_sne, loss] = tsne(data3,'Algorithm','exact','Perplexity',15);
scatter(Yt_sne(:,1),Yt_sne(:,2),sz,c,'filled')

% 5 Diffusion Map
figure()
% estimated epsilon from her heuristic is .184, not enough
eps = .184;
delta = .2;
YDiff = diffMap(data3,eps,delta,expDim);
scatter(YDiff(1,:),YDiff(2,:),sz,c,'filled')




%% problem 2
% S curve with noise
close all
clear


% load data matrix (it's called data3)
load("ScurveData.mat");
[n,~] = size(data3);
expDim = 2;
% for plotting in color (emphasize the change over the s shape to see if it
% folds back)
c = (linspace(1,5,n)+repmat(linspace(1,5,32),1,11))/2;
sz = 25;
% add noise
data3 = data3 + (rand(size(data3))-.5)/2;


% actually run each dimension reduction algorithm
% 1 PCA
% first center the data
Ycentered = data3 - ones(n,1)*sum(data3,1)/n;
[U,Sigma,~] = svd(data3','econ');
Ypca = Ycentered*U(:,1:expDim); % features
scatter(Ypca(:,1),Ypca(:,2),sz,c,'filled')

% 2 Isomap
figure()
Yiso = isomap(data3,10,2);
scatter(Yiso(:,1),Yiso(:,2),sz,c,'filled')

% 3 LLE
figure()
k = 10;                      % # nearest neighbors
Ylle = lle(data3',k,expDim);
scatter(Ylle(1,:),Ylle(2,:),sz,c,'filled')

% 4 t-SNE
figure()
perp = 10;
[Yt_sne, loss] = tsne(data3,'Algorithm','exact','Perplexity',15);
scatter(Yt_sne(:,1),Yt_sne(:,2),sz,c,'filled')

% 5 Diffusion Map
figure()
% estimated epsilon from her heuristic is .184, not enough
eps = .5;
delta = .5;
YDiff = diffMap(data3,eps,delta,expDim);
scatter(YDiff(1,:),YDiff(2,:),sz,c,'filled')

figure()
scatter3(data3(:,1),data3(:,2),data3(:,3),sz,c,'filled');