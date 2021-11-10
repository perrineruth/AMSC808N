function [C,U,R,rel_err,pi_c_adj,pi_r] = CURadj(A,k,c,r,q,alpha)
    % CURADJ
    % combines the distribution of column leverage scores to give a
    % different approximation. Specifically it's the weighted combination
    % of leverage scores and the probability distribution q
    %       p = alpha*q + (1-alpha)*pi_c
    
    % size of A
    [m,n] = size(A);

    % compute SVD and its low rank approximation
    [U,S,V] = svd(A,'econ');
    % convert to rank k SVD
    % Ak = U(:,1:k)*S(1:k,1:k)*V(:,1:k)';
    Uk = U(:,1:k);
    Sk = S(1:k,1:k);
    Vk = V(:,1:k);

    % find the C matrix using columns of A
    % weights, sum rows not cols otherwise all ones
    pi_c = sum(Vk.^2,2)/k;
    pi_c_adj = alpha*q + (1-alpha)*pi_c;
    cols = rand(n,1) < min(1,c*pi_c); % which columns to accept
    C = A(:,cols');

    % next choose R of rows
    pi_r = sum(Uk.^2,2)/k;
    rows = rand(m,1) < min(1,r*pi_r);
    R = A(rows,:);

    % finally use pinv to calculate U
    U = pinv(C)*A*pinv(R);

    rel_err = norm(A-C*U*R,"fro")/norm(A-Uk*Sk*Vk','fro');
end