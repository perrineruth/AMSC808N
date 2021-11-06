function [M,F_err] = NMF_nuclear(A,Ohm,lambda)
    % create approximation matrix
    [n,m] = size(A);
    M = ones(n,m);
    
    % projection function
    P = @(M) M.*Ohm;

    % iteration parameters
    maxIter = 100;
    tol = 1e-4;
    iter = 1;       % initialize
    d = tol+1;
    while iter < maxIter && d > tol
        Mold = M;

        % make the SVD step described
        Err = M + P(A-M);
        [U,S,V] = svd(Err,'econ');
        x = size(S,1); % number singular vecs
        M = U*max(S-lambda*eye(x),0)*V';
        

        d = norm(Mold-M,"fro");
        iter = iter+1;
    end

    F_err= norm(P(A)-P(M),'fro');
end