function [W,H,F_err] = proj_grad(A,k)

    % initialize matrices A=WH and create projection operator
    [m,n] = size(A);
    Pr = @(X) max(X,0); % projection of X to nonegative values
    % initializing to say ones causes an issue of rank deficient
    % W and H matrices in the output (they stay rank 1)
    a = sum(A,'all')/m/n;
    W = randi(5,m,k)/2;
    H = randi(5,k,n)/2; 

    % use R error for computation and step size
    R = A-W*H;
    alpha = @(k) .1/(100); % this works best with decaying step size

    % initialize loop vals
    maxIter = 1e4;  iter = 1;
    tol = 1e-5;     d = tol+1;
    while iter < maxIter && d > tol
        % step matrices
        W = Pr(W + alpha(k)*R*H');
        H = Pr(H + alpha(k)*W'*R);

        
        % loop updates
        iter = iter+1;
        % check avg square element change not too small...
        %d = (norm(Wnew-W,"fro") + norm(Hnew-H,"fro"))/sqrt((n+m)*k);
        %W = Wnew; H = Hnew;
        
        % recompute R
        R = A-W*H;
        d = norm((A-W*H),'fro');
        fprintf('error: %f\n', d^2)
    end

    F_err = norm(A-W*H,"fro");
end