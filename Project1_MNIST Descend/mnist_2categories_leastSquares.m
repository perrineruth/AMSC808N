function mnist_2categories_leastSquares(optimizer,nPCA)
% MNIST_2CATEGORIES_QUADRATIC
%  Inputs:
%   - optimizer = optimizer for quadratic surface, either Gauss Newton
%                   or Levenberg Marquadt
%   - nPCA = number of PCA terms to approximate images
%  Description
%   - creates a quadratic dividing surface of the
%     MNIST dataset to distinguish the char 1 from
%     7.

% close figures & font size
close all
fsz = 20;
%% load in data
mdata = load('mnist.mat');
imgs_train = mdata.imgs_train;
imgs_test = mdata.imgs_test;
labels_test = mdata.labels_test;
labels_train = mdata.labels_train;
%% find 1 and 7 in training data
ind1 = find(double(labels_train)==2);
ind2 = find(double(labels_train)==8);
n1train = length(ind1);
n2train = length(ind2);
fprintf("There are %d 1's and %d 7's in training data\n",n1train,n2train);
train1 = imgs_train(:,:,ind1);
train2 = imgs_train(:,:,ind2);
%% find 1 and 7 in test data
itest1 = find(double(labels_test)==2);
itest2 = find(double(labels_test)==8);
n1test = length(itest1);
n2test = length(itest2);
fprintf("There are %d 1's and %d 7's in test data\n",n1test,n2test);
test1 = imgs_test(:,:,itest1);
test2 = imgs_test(:,:,itest2);
%% use PCA to reduce dimensionality of the problem to 20
[d1,d2,~] = size(train1);
X1 = zeros(n1train,d1*d2);
X2 = zeros(n2train,d1*d2);
for j = 1 : n1train
    aux = train1(:,:,j);
    X1(j,:) = aux(:)';
end
for j = 1 :n2train
    aux = train2(:,:,j);
    X2(j,:) = aux(:)';
end
X = [X1;X2];
D1 = 1:n1train;
D2 = n1train+1:n1train+n2train;
[U,Sigma,~] = svd(X','econ');
esort = diag(Sigma);
figure;
plot(esort,'.','Markersize',20);
grid;
Xpca = X*U(:,1:nPCA); % features
figPCA = figure; 
hold on; grid;
plot3(Xpca(D1,1),Xpca(D1,2),Xpca(D1,3),'.','Markersize',20,'color','k');
plot3(Xpca(D2,1),Xpca(D2,2),Xpca(D2,3),'.','Markersize',20,'color','r');
view(3)
%% split the data to training set and test set
Xtrain = Xpca;
Ntrain = n1train + n2train;
Xtest1 = zeros(n1test,d1*d2);
for j = 1 : n1test
    aux = test1(:,:,j);
    Xtest1(j,:) = aux(:)';
end
for j = 1 :n2test
    aux = test2(:,:,j);
    Xtest2(j,:) = aux(:)';
end
Xtest = [Xtest1;Xtest2]*U(:,1:nPCA);
%% category 1 (1): label 1; category 2 (7): label -1
label = ones(Ntrain,1);
label(n1train+1:Ntrain) = -1;
%% dividing hypersurface: x'*W*x v'*x + b
dim = nPCA;
Y = (label*ones(1,dim + 1)).*[Xtrain,ones(size(Xtrain,1),1)]; 
% Y*w is the test fucntion
%% optimize w and b using a smooth loss function and SINewton

RJ = @(w) Res_and_Jac(Xtrain,label,w);
w = ones(dim^2+dim+1,1);
% deterministic -> no epochs
kmax = 5e1; % the max number of iterations
tol = 1e-3;
n = size(Y,1);
[w,f,gnorm] = optimizer(RJ,w,kmax,tol);
W = reshape(w(1:dim^2),[dim,dim]);
v = reshape(w(dim^2+1:dim^2+dim),[1,dim]);
b = w(end);
% plot the objective function
figure;
plot(f,'Linewidth',2);
xlabel('iter','fontsize',fsz);
ylabel('f','fontsize',fsz);
set(gca,'fontsize',fsz,'Yscale','log');
% plot the norm of the gradient
figure;
plot(gnorm,'Linewidth',2);
xlabel('iter','fontsize',fsz);
ylabel('||g||','fontsize',fsz);
set(gca,'fontsize',fsz,'Yscale','log');

%% apply the results to the test set
Ntest = n1test+n2test;
testlabel = ones(Ntest,1);
testlabel(n1test+1:Ntest) = -1;
test = myquadratic(Xtest,testlabel,w);
hits = find(test > 0);
misses = find(test < 0);
nhits = length(hits);
nmisses = length(misses);
fprintf('n_correct = %d, n_wrong = %d, accuracy %d percent\n', ...
    nhits,nmisses,nhits/Ntest);

%% plot the dividing surface if nPCA = 3
if dim == 3
    xmin = min(Xtrain(:,1)); xmax = max(Xtrain(:,1));
    ymin = min(Xtrain(:,2)); ymax = max(Xtrain(:,2));
    zmin = min(Xtrain(:,3)); zmax = max(Xtrain(:,3));
    nn = 50;
    figure(figPCA);
    MSGID = 'MATLAB:fplot:NotVectorized';
    warning('off', MSGID)
    fimplicit3(@(x,y,z) [x' y' z']*W*[x;y;z]+v*[x;y;z]+b,...
        [xmin xmax ymin ymax zmin zmax])
    p.FaceColor = 'cyan';
    p.EdgeColor = 'none';
    camlight 
    lighting gouraud
    alpha(0.3);
end
end

%% The quadratic approximation
function q = myquadratic(X,y,w)
d = size(X,2);
d2 = d^2;
W = reshape(w(1:d2),[d,d]);
v = w(d2+1:d2+d);
b = w(end);
qterm = diag(X*W*X');
q = y.*qterm + ((y*ones(1,d)).*X)*v + y*b;
end


