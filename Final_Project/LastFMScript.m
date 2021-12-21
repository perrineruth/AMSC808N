%% Script to run final project analysis
% open LastFM (from SNAP) data and create sparse adjacency matrix
% 
% data from website
% number of nodes: 7624
% edge density: 0.001 (pretty sparse)
close all
clear all


n = 7624;                               % #nodes
E = readmatrix('lastfm_asia_edges.csv');     % edges
A = sparse(E(:,1)+1,E(:,2)+1,1,n,n);    % adjacency, note change index 0->1
A = A+A';

%% find the number of connected components
% using the old BFS code find connected components

s = BFS(A); % this gives that the largest CC is as big as our graph...
fprintf(['\nsize of largest connected component: %d\n' ...
         'number of nodes:                     %d\n'],max(s),n)
% our graph is connected

%% Find the degree distribution

k = full(sum(A,1)); % degrees, this is dense so why sparse is silly here...

% find pmf for degree distribution
kmax = max(k);
p = zeros(1,kmax); % all nodes are at least degree 1, don't need 0
for i = 1:kmax
    p(i) = sum(k==i)/n;
end

% find the largest n such that 2^(n+1)<=kmax
nmax = floor(log(kmax)/log(2)) - 1;

% bin using log binning
pbin = zeros(1,nmax+1);
kbin = zeros(1,nmax+1);

for i = 0:nmax
    % expected degree for a bin
    kvals = (2^i):(2^(i+1)-1);
    kbin(i+1) = sum(p(kvals) .* (kvals)) / sum(p(kvals));
    % binned pmf (avg probability)
    pbin(i+1) = sum(p(kvals))/2^i;
end

% saturation value
ksat = 0;

loglog((1:kmax)+ksat,p,'b.','MarkerSize',12)
hold on
loglog(kbin+ksat,pbin,'r.','MarkerSize',18)



%% solve linear least squares problem setup in paper

nmin = 0;
nvals = (nmin+1):(nmax+1);
kvals = kbin(nvals)+ksat;
M = [ones(nmax-nmin+1,1), -kvals', -log(kvals)'];
b = log(pbin(nvals))';


x = M\b;

C = exp(x(1));
a = x(2);
r = x(3);

papprox = @(k)C*exp(-a*k).*k.^(-r);
C = C/sum(papprox(1:1e4));
loglog(kbin+ksat,papprox(kbin+ksat),'--','LineWidth',2)

% to make a nice figure
fs = 20;
xlabel('degree (k+k_{sat})','FontSize',fs)
ylabel('probability (p_k)','FontSize',fs)
legend('degree distribution','binning data','truncated power law')


%% solve for average shortest path length

% create a graph from adjacency matrix
G = graph(A);
% distance matrix
D = distances(G);
% D(i,j) distance i to j. Distances repeated twice (n(n-1) nonzero entries)
x = sum(D,'all')/(n*(n-1));
fprintf('\nAverage Distance: %f\n', x)

% G0'(1)
kTh = 1:kmax;
pkTh = C* exp(-a*kTh).*kTh.^(-r);
z1 = sum(kTh.*pkTh);
kTh = 2:kmax;
% z2 =G0''(1)
pkTh = C* exp(-a*kTh).*kTh.^(-r);
z2 = sum(kTh.*(kTh-1).*pkTh);

% estimated average path length
l = log(n/z1)/log(z2/z1)+1;

%% compute number of paths

paths = 0;
Cpaths = 0;

% look at paths uvw
for v = 1:n
    N = find(A(v,:)); % neighbors of v
    for u = N
        for w = N
            % make sure not a path of 1 edge
            if u~=w
                paths = paths+1;
                if A(u,w)
                    Cpaths = Cpaths+1;
                end
            end
        end
    end
end

Clustering = Cpaths/paths;
fprintf('\nClustering Coeff: %f\n',Clustering)

% compute the random clustering coefficient
km = sum((1:kmax).*p);
km2 = sum((1:kmax).^2 .* p);
Cr = (km2-km)^2/(n*km^3);
fprintf('Random Clustering Coeff: %f\n',Cr)

%% critical transmissibility

kvals = 1:kmax;
Tc = sum(kvals.*p)/sum(kvals.*(kvals-1).*p);
fprintf('\n Critical Transmissibility: %f\n',Tc)

%% simulate SIR Model

close all

T=.4;
% transmission edges
Et = E(rand(27806,1)<T,:);

% Adjacency matrix for transmission network
At = sparse(Et(:,1)+1,Et(:,2)+1,1,n,n);   
At = At+At';

% S I R states
S = ones(1,n);
I = zeros(1,n);
R = zeros(1,n);

% Variables to be plotted
steps = 0:10;
Infected = zeros(1,11);
Infected(1) = 1; % start one infected

u0 = randi(n); % source of the epidemic
I(u0) = 1;
S(u0) = 0;

for i = 1:10
    CurInf = find(I); % infected nodes start of timesteps
    I(CurInf) = 0;
    R(CurInf) = 1; % recover nodes
    for u = CurInf
        N = find(At(u,:)); % neighbors of infected node on Transmission graph
        N = N(S(N)==1); %only take the susceptable neighbors
        I(N) = 1;
        S(N) = 0;
    end
    Infected(i+1) = sum(I);
end

plot(steps,Infected);

sum(Infected)/n
%% 
% transition given by Newman
real(polylog(r-1,exp(-a))/(polylog(r-2,exp(-a))-polylog(r-1,exp(-a))))

pl = @(r,x) real(polylog(r,x));

G0 = @(x) pl(r,x*exp(-a))/pl(r,exp(-a));
G1 = @(x) (pl(r-1,x*exp(-a))./(x*pl(r-1,exp(-a))));

u = fsolve(@(x) x - G1(1+(x-1)*T),.25);

S = 1-G0(1+(u-1)*T);

kap = km2/km;

pc = 1-1/(kap-1);
% kappa infection network
kapi = 1+ T*(km2-km)/km;
pci = 1-1/(kapi-1);