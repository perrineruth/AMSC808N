function [W,H,F_err] = LeeSeung(A,k)
    
    % initialize matrices
    [m,n] = size(A);
    a = sum(A,'all')/m/n;
    % tend to be better behaved when closer to average A value and make
    % sure values in W,H not to close to 0
    % initialization seems difficult and this appears to help
    W = (rand(m,k)+.5)*sqrt(a/k);
    H = (rand(k,n)+.5)*sqrt(a/k); 

    % initialize loop vals
    maxIter = 1e5;  iter = 1;
    tol = 1e-5;     d = tol+1;
    while iter < maxIter && d > tol
        % step sizes
%         S = W./(W*(H*H'));
%         Sp = H./((W'*W)*H);

        % new approximation matrices
        Wnew = (W .* (A*H')) ./ (W*(H*H'));
        Hnew = (H .* (W'*A)) ./ ((W'*W)*H);

        % update loop params
        d = (norm(Wnew-W,"fro") + norm(Hnew-H,"fro"))/sqrt((n+m)*k);
        iter = iter+1;
        W=Wnew; H=Hnew;
    end

    % final frobenius error
    F_err = norm(A-W*H,"fro");
end