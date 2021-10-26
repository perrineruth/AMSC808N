function [w,f,normgrad] = NAG(fun,gfun,Hvec,n,w,bsz,kmax,tol)

% initial guess w training 
I = 1:n;                % list of indices for convenience
f = zeros(kmax + 1,1);  % values of loss func.
f(1) = fun(I,w);        % loss initially
normgrad = zeros(kmax,1); 

% setup for Nesterov gradient descent
u = @(k) 1-3/(5+k);
a = @(k) .25;
y = w; % start y = w

% when to evaluate fun only do it at 1.5^p timesteps to save computation
% time while still getting some precise measurements...
p=0;

for k = 1 : kmax
    % random batch and gradient
    Ig = randperm(n,bsz);
    b = gfun(Ig,w);         % for measurement but not computation
    gy = gfun(Ig,y);        % grad y for computation
    normgrad(k) = norm(b);

    % update parameters
    wnew = w - a(k)*gy;
    y = (1+u(k))*wnew - u(k)*w;
    w = wnew;
    
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
        
        
    
    
