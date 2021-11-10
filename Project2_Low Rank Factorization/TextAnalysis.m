%% (3) first find common words and evaluate 
clear all
clf

[M,~,y,words] = readdata();
pi_c = zeros(19,size(M,2));
pi_r = zeros(19,size(M,1));

for k = 2:20
    c = k;      
    r = k;
    [~,~,~,~,pi_c(k-1,:),pi_r(k-1,:)] = CUR(M,k,c,r);
 
%     figure(1)
%     hold on
%     plot(pi_c);
% 
%     figure(2)
%     hold on
%     plot(pi_r);
end
figure(1)
plot(1:size(M,2),pi_c)
title('Column Leverage Scores','FontSize',20)
legend('k=2','k=3','k=4','k=5','k=6','k=7','k=8','k=9','k=10','k=11',...
    'k=12','k=13','k=14','k=15','k=16','k=17','k=18','k=19','k=20')

figure(2)
plot(1:size(M,1),pi_r)
title('Row Leverage Scores','FontSize',20)
legend('k=2','k=3','k=4','k=5','k=6','k=7','k=8','k=9','k=10','k=11',...
    'k=12','k=13','k=14','k=15','k=16','k=17','k=18','k=19','k=20')


%% generate figure of mean ratio for fixed k

% fix k and evaluate for various c and r
k = 10;
cr_draw = [.5 .75 1 1.25 1.5 1.75 2]*k; l=length(cr_draw);
Rel_err = zeros(l,l);
[Cvals,Rvals] = meshgrid(cr_draw,cr_draw);
for i = 1:l
    for j = 1:l
        rel_err = 0;
        Trials = 20;
        for iter = 1:Trials
            c = cr_draw(i);
            r = cr_draw(j);
            [C,U,R,rel_err,pi_c,pi_r] = CUR(M,k,c,r);
            Rel_err(i,j) = Rel_err(i,j)+rel_err;
            %fprintf('relative error for c=%f: %f',c,Rel_err)
        end
        Rel_err(i,j) = Rel_err(i,j)/Trials;
    end
end

figure(3)
heatmap(Rel_err,'XData',cr_draw,'YData',cr_draw,'Title','k=20','FontSize',20);
xlabel('c');ylabel('r')
axes equal


%% Problem 4
% find Information gain and turn into a probability mass function for cols

[M,Mf,y,words] = readdata();
[n,d] = size(M);
% indices for each dataset
i1 = find(y==-1);
i2 = find(y==1);
ii = find(M>0);
n1 = length(i1);
n2 = length(i2);

M = full(M);
Mfreq = sum(M,1)/n;
M1freq = sum(M(i1,:),1)/n1;
M2freq = sum(M(i2,:),1)/n2;
% information gain, Mfreq not included in paper, but it doesn't matter
% since it gets normalized anyways
IG = abs(M1freq-M2freq);%.*Mfreq;
q = IG'/sum(IG);                 % information gain as a pmf for cols

% fix k=c=r and contaminate the original col levarage scores w/ our new
% distribution as alpha*q + (1-alpha)*column_leverage_score

% find alpha so following I2 included in I1
alpha = .075;
% only want to check probability distribution is adjusted as desired...
k=20; c=2*k; r=2*k;
[~,~,~,~,pi_c_adj,~] = CURadj(M,k,c,r,q,alpha);
[~,I1] = maxk(pi_c_adj,20); % 20 top adjusted leverage scores
[~,I2] = maxk(q,10);         % top 5 information gain indices

% Note, stop words not coded into this .m file (see William's code, it
% takes significant effort and is additional to what's given in the
% project.

% convert top 5 words for pi_c_adj to PCA:
ind5 = I1(1:5);

C = M(:,ind5); 
[U, S, ~] = svd(C);
for i = 1:length(y)
    if y(i) ==1

        break;
    end
end
%figure(1)
P = U(:,1:2)*S(1:2,1:2);
plot(P(i1,1), P(i1,2),'.','Markersize',20,'color','k')
hold on 
%figure(2)
plot(P(i2,1), P(i2,2),'bo','Markersize',8,'LineStyle','none')
legend("Indiana","Florida")
