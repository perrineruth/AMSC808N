function [w,f,normgrad] = LM(RJ,w,kmax,tol)

% trust region params
DMax = 1;
Dmin = 0.00; %min max radii
D = 1.17;  % current tr
nu = .1;


% initial guess w, evaluating initial approx
[r,J] = RJ(w);
f = zeros(kmax + 1,1);  % values of loss func.
f(1) = r'*r/2;        % loss initially
normgrad = zeros(kmax,1); 


for k = 1 : kmax
    
    H = J'*J + eye(length(w))*1e-6;
    g = J'*r;
    p = -H\g;

    if norm(p) > D
        p = constrained(H,g,D);
    end
    wnew = w+p;
    [rnew,Jnew] = RJ(wnew);
    fnew = rnew'*rnew/2;
    rho = -(f(k) - fnew)/(p'*g + p'*H*p/2);
    if rho < 0.25
        D = max(Dmin,.25*D);
    else
        if rho > 0.75 && abs(norm(p)-D) == 1e-10
            D = min(2*D,DMax);
        end
    end
    if rho > nu
        w = wnew;
        f(k+1) = fnew;
        r = rnew;
        J = Jnew;
    else
        f(k+1) = f(k);
    end

    normgrad(k) = norm(g);

   
    if mod(k,1)==0
        fprintf('k = %d, f = %d, ||g|| = %d\n',k,f(k+1),normgrad(k));
    end

    if normgrad(k) < tol
        break;
    end
end
fprintf('k = %d, f = %d, ||g|| = %d\n',k,f(k+1),normgrad(k));
end
        
%% methods for solving constrained optimization

function p = constrained(H,g,D)
    lam = 1;
    n = size(H,2);
    while 1
        H1 = H+lam*eye(n);
        C = chol(H1);
        p = -C\(C'\g);
        np = norm(p);
        dd = abs(np - D);
        if dd < 1e-6
            break
        end
        
        q = C' \p;
        nq = norm(q);
        lamnew = lam + (np/nq)^2 *(np-D)/D;
        if lamnew < 0
            lam = 0.5*lam;
        else
            lam = lamnew;
        end
    end
end
