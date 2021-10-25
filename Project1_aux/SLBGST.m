function [w,f,normgrad] = SLBGST(fun,gfun,Hvec,n,w,bsz,kmax,tol)
% SLBGST
%   Stochastic L-BFGS


% initial guess w, evaluating initial approx 
I = 1:n;                % list of indices for convenience
f = zeros(kmax + 1,1);  % values of loss func.
f(1) = fun(I,w);        % loss initially
normgrad = zeros(kmax,1); 

% Implimentation parameters for SLBGST
bszh = bsz*10; % sample size for Hessian sampling
HM = 20;    % frequency of updating the Hessian inverse
m = 5;      % number of steps for generating approximation
HIter = 0;  % number of evaluation of s and y computed

% parameters for evaluating the approx of inverse Hessian
s = zeros(length(w),m);
y = zeros(length(w),m);
rho = zeros(1,m);

% initial step gradient descent for computing H properly
Ig = randperm(n,bsz);
Ih = randperm(n,bszh);
g = gfun(Ig,w);
gh = gfun(Ih,w);    % stored gradient with larger batch size
wh = w;             % keeping track of old w values

w = w-.1*g;
Ih = randperm(n,bszh);
ghnew = gfun(Ih,w);

s(:,1) = w-wh;
y(:,1) = ghnew-gh;
HIter = 1;
gh = ghnew;
wh = w;
rho(1) = 1/(s(:,1)'*y(:,1));
jmax=10;
eta=.5;
for k = 1 : kmax
    % random batch and gradient
    Ig = randperm(n,bsz);
    g = gfun(Ig,w);         % for measurement but not computation
    p = finddirection(g,s,y,rho,min(HIter,m));
   % if any(rho(2:5))
        %fprintf("here")
   % end
    normgrad(k) = norm(g);

    
    [a,j] = linesearch(w,p,g,fun,eta,Ig,jmax);
    if j == jmax 
        p=-g;
        [a,j] =  linesearch(w,p,g,fun,eta,Ig,jmax);
    end
    w = w + a*p;

    % if we need to update Hessian
    if mod(k,HM) == 0 || HIter < m
        s = circshift(s,[0,1]);
        y = circshift(y,[0,1]);
        rho = circshift(rho,[0,1]);
        
        Ih = randperm(n,bszh);
        ghnew = gfun(Ih,w);
        s(:,1) = w-wh;
        y(:,1) = ghnew-gh;
        HIter = HIter+1;
        gh = ghnew;
        wh = w;
        rho(1) = 1/(s(:,1)'*y(:,1));
    end


    f(k+1) = fun(Ig,w);
   
    if mod(k,100)==0
        fprintf('k = %d, f = %d, ||g|| = %d\n',k,f(k+1),normgrad(k));
    end

    if normgrad(k) < tol
        break;
    end
end
fprintf('k = %d, f = %d, ||g|| = %d\n',k,f(k+1),normgrad(k));
end
        
%% methods for line search for SLBGST
% finding direction from Hessian approx so far

function p = finddirection(g,s,y,rho,m)
    
    a = zeros(m,1);
    for i=1:m
        a(i) = rho(i)*s(:,i)'*g;
        g = g-a(i)*y(:,i);
    end
    gam = (s(:,1)'*y(:,1))/(y(:,1)'*y(:,1));
    g = g*gam;
    for i = m:-1:1
        aux = rho(i)*y(:,i)'*g;
        g = g+(a(i)-aux)*s(:,i);
    end
    p = -g;
end
    
% line search method on a given direction
function [a,j] = linesearch(x,p,g,func,eta,If,jmax)
    gam = 0.9;

    a = 1;
    f0 = func(If,x);
    aux = eta*g'*p;
    for j = 0:jmax
        xtry = x+a*p;
        f1 = func(If,xtry);
        if f1 <f0+aux
            break
        else
            a = a*gam;
        end
    end
end
