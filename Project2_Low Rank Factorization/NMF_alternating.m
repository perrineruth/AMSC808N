function [X,Y,F_err] = NMF_alternating(A,Ohm,k,lambda)
    %% feed in A matrix and tikhonov lambda
    % default max number of iterations
    maxIter = 100;
    tol = 1e-4;

    % projection onto Omega
    P = @(M) M.*Ohm;

    [m,n] = size(A);
    rl = sqrt(lambda);
    Ohm = rand(m,n)>.2;
    [m,n] = size(A);
    X = ones(m,k);
    Y = ones(n,k);
    
    % use alternating iteration to generate low-rank approximation
    iter = 1;
    d = tol+1;
    while iter < maxIter && d > tol
        Xold = X;
        Yold = Y;
        % update X matrix
        for i = 1:m
            M = [Y(Ohm(i,:)==1,:);
                rl*eye(k)];
            b = [A(i,Ohm(i,:)==1),zeros(1,k)]';
            X(i,:) = (M\b)';
        end
    
        % update Y matrix
        for j = 1:n
            M = [X(Ohm(:,j)==1,:);
                rl*eye(k)];
            b = [A(Ohm(:,j)==1,j);zeros(k,1)];
            Y(j,:) = (M\b)';
        end

        d = norm(Xold*Yold'-X*Y','fro');
        iter = iter+1;
    end 
    
    F_err = norm(P(A-X*Y'),'fro');
end