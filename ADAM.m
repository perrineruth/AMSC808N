function [w,f,normgrad] = ADAM(fun,gfun,Hvec,n,w,bsz,kmax,tol)


% initial guess w training 
I = 1:n;                % list of indices for convenience
f = zeros(kmax + 1,1);  % values of loss func.
f(1) = fun(I,w);        % loss initially
normgrad = zeros(kmax,1); 

% setup for Nesterov gradient descent
m=0;
v=0;
% parameters
B1 = 0.2;
B2 = 0.9;
eps = 1e-8;
nu = 0.005;

% when to evaluate fun only do it at 1.5^p timesteps to save computation
% time while still getting some precise measurements...
p=0;

for k = 1 : kmax
    % random batch and gradient
    Ig = randperm(n,bsz);
    g = gfun(Ig,w);         % for measurement but not computation
    g2 = g.^2;              % grad squared for 2nd moment
    normgrad(k) = norm(g);

    % update moments
    m = B1*m + (1-B1)*g;
    v = B2*v + (1-B2)*g2;
    % remove bias
    mh = m/(1-B1^(k+1));
    vh = v/(1-B2^(k+1));
    % finally update state
    w = w - nu.*mh./(sqrt(vh)+eps);
    
    if any(vh<0)
        break
    end

    f(k+1) = fun(Ig,w);
    % to increase accuracy:
%     If = randperm(n,bsz*5);
%     f(k+1) = fun(If,w);
%     b = gfun(If,w);
%     normgrad(k) = norm(b);

    if mod(k,100)==0
        fprintf('k = %d, f = %d, ||g|| = %d\n',k,f(k+1),normgrad(k));
    end

    if normgrad(k) < tol
        break;
    end
end
fprintf('k = %d, f = %d, ||g|| = %d\n',k,f(k+1),normgrad(k));
end
        
        
    
    
