function [Psi] = diffMap(X,eps,delta,dim)
    % Compute the square distance matrix
    % rows of X are data
    [n,~] = size(X);
    D2 = zeros(n);
    for i = 1:n
        for j= 1:n
            D2(i,j) = norm(X(i,:) - X(j,:))^2;
        end
    end

    % Diffusion kernel
    K = exp(-D2/eps);
    q = sum(K,2);       % row sums of K
    P = diag(q)\K;      % prob transition matrix
    pi = q/sum(q);      % invariant distribution

    % solving for R where P = R*lam*L = R*lam*R'*diag(pi)
    % want right eigenvectors
    rtP = diag(pi)^(1/2); 
    rtPinv = rtP^-1;
    [V,lam] = eig(rtP * P * rtPinv);
    % sort eigenvalues from 1 descending
    [absevals,ind] = sort(abs(diag(lam)),'descend');
    V = V(:,ind);
    lam = lam(ind,ind);

    R = rtPinv * V;

    % ignore first eigenvector
    R1 = R(:,2:(dim+1));
    t = ceil( log(1/delta) / log(absevals(2)/absevals(dim+1)) );
    lam1t = lam(2:(dim+1),2:(dim+1)).^t;
    
    % embedding is 
    Psi = (R1*lam1t)';
end