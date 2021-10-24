function [w,f,normgrad] = NAG(fun,gfun,Hvec,n,w,bsz,kmax,tol)

% initial guess w training 
I = 1:n;                % list of indices for convenience
f = zeros(kmax + 1,1);  % values of loss func.
f(1) = fun(I,w);        % loss initially
normgrad = zeros(kmax,1); 

% setup for Nesterov gradient descent
u = @(k) 1-3/(5+k);
a = @(k) .05;
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
    
%     f(k+1) = fun(Ig,w);
    if k>=(21/20)^p
        f(k+1) = fun(I,w);
%         normgrad(k) = norm(gfun(I,w));
        while k>(21/20)^p
            p=p+1;
        end
    elseif mod(k,500)==0
        f(k+1) = fun(I,w);
%         normgrad(k) = norm(gfun(I,w));
    else
        f(k + 1) = f(k);
%         normgrad(k) = normgrad(k-1);
    end
    if mod(k,100)==0
        fprintf('k = %d, f = %d, ||g|| = %d\n',k,f(k+1),normgrad(k));
    end

    if normgrad(k) < tol
        break;
    end
end
fprintf('k = %d, f = %d, ||g|| = %d\n',k,f(k+1),normgrad(k));
end
        
        
    
    
