function [C,U,R] = CUR(A,k)

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
    c = k;     % want O(klogk\eps^2)
    % weights, sum rows not cols otherwise all ones
    pi = sum(Vk.^2,2)/k;
    cols = rand(n,1) < min(1,c*pi); % which columns to accept
    C = A(:,cols');

    % next choose R of rows
    r = c; % not sure where I would change this up...
    pi = sum(Uk.^2,2)/k;
    rows = rand(m,1) < min(1,c*pi);
    R = A(rows,:);

    % finally use pinv to calculate U
    U = pinv(C)*A*pinv(R);


end