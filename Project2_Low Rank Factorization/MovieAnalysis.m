%% Script for interpreting class data of movie ratings
% first read the data from the CSV file
Table = readtable('MovRankData.csv');
A = Table{:,2:end};     % data, first column is user ID
Ohm = isnan(A)==0;        % valid data points
A(isnan(A)==1) = 0;        % so that we get 0*NaN=0...

%% part (1a)
% Alternating Iteration low rank approximation
k=5;
%for l = [.01 .05 .1 .25 .5 1 10 100]
[W,H,err] = NMF_alternating(A,Ohm,k,.1);
err
%end

%% part (1b)
% Approximation with Nuclear norm
[M,err] = NMF_nuclear(A,Ohm,.1);
err

%% part (2a)
% projected gradient descend on the matrix from 1b
k=3;

[W,H,F_err] = proj_grad(M,k);
F_err

%% (2b)


