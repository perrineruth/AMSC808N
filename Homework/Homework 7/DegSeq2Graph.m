function A = DegSeq2Graph(degDist, n)
    k = degDist(n); % generate degrees for n nodes
    S = sum(k); % number of stubs
    while mod(S,2) ~= 0
        k = degDist(n);
        S = sum(k);
    end

    Stubs = zeros(1,S);
    index = 1;
    for i = 1:n
        Stubs(index:index+k(i)-1) = i; % labels for stubs
        index = index+k(i);
    end

    % shuffle stubs
    Stubs = Stubs(randperm(S));
    % pair by splitting vector in 2
    rows = Stubs(1:S/2);
    cols = Stubs(S/2+1:end);


    % this automatically ignores repeats to get an adjacency matrix
    A = sparse(rows,cols,1,n,n);
    A = A+A'; % symmetric

    % remove repeated edges
    A = min(A,1);
    % find diagonal entries make diagonal matrix and subtract
    A = A - diag(diag(A));
end