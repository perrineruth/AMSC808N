function [w,f,normgrad] = Gauss_Newton(RJ,w,kmax,tol)


% initial guess w, evaluating initial approx
[r,J] = RJ(w);
f = zeros(kmax + 1,1);  % values of loss func.
f(1) = r'*r/2;        % loss initially
normgrad = zeros(kmax,1); 

% parameters for line search
jmax=10;
eta=.5;

for k = 1 : kmax
    
    H = J'*J + eye(length(w))*1e-6;
    g = J'*r;
    p = -H\g;

    normgrad(k) = norm(g);

    
    [a,j,rtry,Jtry,f(k+1)] = linesearch(w,p,g,eta,jmax,f(k),RJ);
    if j == jmax 
        p=-g;
        [a,~,rtry,Jtry,f(k+1)] =  linesearch(w,p,g,eta,jmax,f(k),RJ);
    end
    r=rtry;J=Jtry;
    w = w + a*p;

   
    if mod(k,1)==0
        fprintf('k = %d, f = %d, ||g|| = %d\n',k,f(k+1),normgrad(k));
    end

    if normgrad(k) < tol
        break;
    end
end
fprintf('k = %d, f = %d, ||g|| = %d\n',k,f(k+1),normgrad(k));
end
        
%% methods for line search for Gauss Newton
    
% line search method on a given direction
% already compute error, next residuals and Jac so let's use it
function [a,j,r,J,f1] = linesearch(x,p,g,eta,jmax,f0,RJ)
    gam = 0.9;

    a = 1;
    aux = eta*g'*p;
    for j = 0:jmax
        xtry = x+a*p;
        [r,J] = RJ(xtry);
        f1 = r'*r/2;
        if f1 <f0 + a*aux
            break
        else
            a = a*gam;
        end
    end
end
