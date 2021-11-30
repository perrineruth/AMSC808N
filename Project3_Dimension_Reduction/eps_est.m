function eps = eps_est(data)
    % compute square dist matrix    
    [n,~] = size(data);
    D2 = zeros(n);
    for i = 1:n
        for j= 1:n
            D2(i,j) = norm(data(i,:) - data(j,:))^2;
        end
    end

    drowmin = zeros(n,1);
    for i = 1:n
        drowmin(i) = min(D2(i,setdiff(1:n,i)));
    end
    eps = 2*mean(drowmin);
end