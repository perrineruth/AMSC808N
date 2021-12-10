%% Problem 1.c find component ratios
G0 = @(z,p) p*z + (1-p)*z.^3;
km = @(p) 3-2*p;
H1 = @(z,p) ( km(p) - sqrt(km(p)^2-12*z.^2*p*(1-p)) ) ./ (6*(1-p)*z);
H0 = @(z,p) z.*G0(H1(z,p),p);

%% 1c
p = .875;
Ps = zeros(1,31);
for i = 0:30
    Ps(i+1) = GFInt(@(z) H0(z,p),i,100);
end
plot(0:30,Ps,'.','MarkerSize',20)
xlabel('s')
ylabel('Ps')

%% 1d
p = .4;
Ps = zeros(1,31);
for i = 0:30
    Ps(i+1) = GFInt(@(z) H0(z,p),i,100);
end
plot(0:30,Ps,'.','MarkerSize',20)
xlabel('s')
ylabel('Ps')

%% 2a
p = .875;
n = 1e4;
degDist = @(n) 1+2*(rand(1,n) > p);
A = DegSeq2Graph(degDist,n);

CompSize = BFS(A,n);

PsExp = zeros(1,50);
PsTh = zeros(1,50);
for i = 1:50
    PsExp(i) = sum(CompSize==i);
    PsTh(i) = GFInt(@(z) H0(z,p),i,100);
end
PsExp = PsExp/sum(PsExp);
plot(1:50,PsTh,'b.',1:50,PsExp,'r.','MarkerSize',16)
xlabel('s')
ylabel('Ps')
legend('Theoretical', 'Experimental','FontSize',20)

%% 2b
p = .4;
n = 1e4;
degDist = @(n) 1+2*(rand(1,n) > p);
A = DegSeq2Graph(degDist,n);

CompSize = BFS(A,n);

PsExp = zeros(1,50);
PsTh = zeros(1,50);
for i = 1:50
    PsExp(i) = sum(CompSize==i);
    PsTh(i) = GFInt(@(z) H0(z,p),i,100);
end
PsExp = PsExp/n;
plot(1:50,PsTh,'b.',1:50,PsExp,'r.','MarkerSize',16)
xlabel('s')
ylabel('Ps')
legend('Theoretical', 'Experimental','FontSize',20)

%% find size of giant CC in prev.
m = max(CompSize); % number of nodes in giant CC
S = m/n;