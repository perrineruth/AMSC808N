function [W,H,F_err] = proj_grad(A,k)

    % initialize matrices A=WH and create projection operator
    [m,n] = size(A);
    Pr = @(X) max(X,0); % projection of X to nonegative values
    % initializing to say ones causes an issue of rank deficient
    % W and H matrices in the output (they stay rank 1)
    W = rand(m,k);
    H = rand(k,n);

    % use R error for computation and step size
    R = A-W*H;
    alpha = @(k) 1/(k+10); % this works best with decaying step size

    % initialize loop vals
    maxIter = 100;  iter = 1;
    tol = 1e-4;     d = tol+1;
    while iter < maxIter && d > tol
        % step matrices
        Wnew = Pr(W + alpha(k)*R*H');
        Hnew = Pr(H + alpha(k)*W'*R);

        % recompute R
        R = A-W*H;

        % loop updates
        iter = iter+1;
        d = norm(Wnew-W,"fro") + norm(Hnew-H,"fro");
        W = Wnew; H = Hnew;
    end

    F_err = norm(A-W*H,"fro");
end