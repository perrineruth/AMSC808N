function s =BFS(A,n)
    Available = ones(1,n);
    Explored = zeros(1,n);
    s = zeros(1,n);
    Queue = [];
    index = 1;

    while any(Available)
        v = find(Available,1);
        
        Available(v) = 0;
        Explored(v) = 1;
        for i = 1:n
            if A(v,i) && Available(i)
                Queue = [Queue i]; % sorry for poor allocation its faster to write
                Available(i) = 0;
            end
        end

        while ~isempty(Queue)
            v = Queue(1);
            Queue(1) = [];
            Available(v) = 0;
            Explored(v) = 1;
            for i = 1:n
                if A(v,i) && Available(i)
                    Queue = [Queue i]; % sorry for poor allocation its faster to write
                    Available(i) = 0;
                end
            end
        end
        s(Explored==1) = sum(Explored);
        Explored= zeros(1,n);
    end
end