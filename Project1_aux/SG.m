function [w,f,normgrad] = SG(fun,gfun,Hvec,n,w,bsz,kmax,tol)
alpha = @(k) 80/(200+k);

% initial guess w training 
I = 1:n;
f = zeros(kmax + 1,1); % values of loss func.
f(1) = fun(I,w);
normgrad = zeros(kmax,1);

% when to evaluate fun only do it at 1.5^p timesteps
p=0;

for k = 1 : kmax
    % random batch and gradient
    Ig = randperm(n,bsz);
    b = gfun(Ig,w);
    normgrad(k) = norm(b);

    % update parameters
    w = w - alpha(k)*b; 
    
    if k>(21/20)^p
        f(k+1) = fun(I,w);
        while k>(21/20)^p
            p=p+1;
        end
    elseif mod(k,500)==0
        f(k+1) = fun(I,w);
    else
        f(k + 1) = f(k);
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
        
        
    
    
