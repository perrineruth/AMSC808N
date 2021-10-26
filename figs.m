function figs()
% Generate figures for MNIST quadratic approximations
nPCA = 20;

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
% figure;
% plot(esort,'.','Markersize',20);
% grid;
Xpca = X*U(:,1:nPCA); % features
% figPCA = figure; 
% hold on; grid;
% plot3(Xpca(D1,1),Xpca(D1,2),Xpca(D1,3),'.','Markersize',20,'color','k');
% plot3(Xpca(D2,1),Xpca(D2,2),Xpca(D2,3),'.','Markersize',20,'color','r');
% view(3)


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
lam = 0.001; % Tikhonov regularization parameter
fun = @(I,w)qloss(I,Xtrain,label,w,lam);
gfun = @(I,w)qlossgrad(I,Xtrain,label,w,lam);
Hvec = @(I,w,v)Hvec0(I,Y,w,v,lam);
w = ones(dim^2+dim+1,4);
% params for epochs and batchsize
bsz = 256;
%frac = 100;
frac= Ntrain/bsz; % batch size
kmax = floor(2e1*frac); % the max number of iterations
tol = 1e-4;
n = size(Y,1);
% figures for plotting


% run through optimizers
optimizers = {@SG @NAG @ADAM @SLBGST};

f = zeros(kmax+1,4);
gnorm = zeros(kmax,4);
for i = 1:4
    opt = optimizers{i};
    [w(:,i),f(:,i),gnorm(:,i)] = opt(fun,gfun,Hvec,n,w(:,i),bsz,kmax,tol);
end

figure(1);
plot(f)
xlabel('iter','fontsize',fsz);
ylabel('f','fontsize',fsz);
set(gca,'fontsize',fsz,'Yscale','log');
legend('SG','NAG','ADAM','SLBGST')


figure;
plot(gnorm)
xlabel('iter','fontsize',fsz);
ylabel('||g||','fontsize',fsz);
set(gca,'fontsize',fsz,'Yscale','log');
legend('SG','NAG','ADAM','SLBGST')


%% apply the results to the test set
for i = 1:4
    Ntest = n1test+n2test;
    testlabel = ones(Ntest,1);
    testlabel(n1test+1:Ntest) = -1;
    test = myquadratic(Xtest,testlabel,1:Ntest,w(:,i));
    hits = find(test > 0);
    misses = find(test < 0);
    nhits = length(hits);
    nmisses = length(misses);
    fprintf('n_correct = %d, n_wrong = %d, accuracy %d percent\n', ...
        nhits,nmisses,nhits/Ntest);
end
end

%%
function g = qlossgrad(I,Xtrain,label,w,lam)
aux = exp(-myquadratic(Xtrain,label,I,w));
a = -aux./(1+aux);
X = Xtrain(I,:);
d = size(X,2);
d2 = d^2;
y = label(I);
ya = y.*a;
qterm = X'*((ya*ones(1,d)).*X);
lterm = X'*ya;
sterm = sum(ya);
g = [qterm(:);lterm;sterm]/length(I) + lam*w;
end
%%
function q = myquadratic(Xtrain,label,I,w)
X = Xtrain(I,:);
d = size(X,2);
d2 = d^2;
y = label(I);
W = reshape(w(1:d2),[d,d]);
v = w(d2+1:d2+d);
b = w(end);
qterm = diag(X*W*X');
q = y.*qterm + ((y*ones(1,d)).*X)*v + y*b;
end


