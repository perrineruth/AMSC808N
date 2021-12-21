function s = BFS(A)
% BFS - Breadth First Search
%   Performs BFS on a given graph to find its connected components
%   input:
%    A - Adjacency matrix
%   output:
%    s - vector where s(i) is the size of the CC that contains node i

    n = size(A,1); % number of nodes assuming A symmetric

    Available = ones(1,n);  % unexplored nodes
    Explored = zeros(1,n);  % traversed nodes in the current connected component
    s = zeros(1,n);         % size of connected node that holds node i as a vector
    
    Queue = [];
    
    while any(Available)
        % initialize connected component
        v = find(Available,1);
        
        Available(v) = 0;
        Explored(v) = 1;
        N = find(A(v,:)); % vectorized instead of in the for loop, much faster...
        for u = N
            if Available(u)
                Queue = [Queue u]; % sorry for poor allocation its faster to write
                Available(u) = 0;
            end
        end

        while ~isempty(Queue)
            % repeat above but in the FIFO queue
            v = Queue(1);
            Queue(1) = [];
            Available(v) = 0;
            Explored(v) = 1;
            N = find(A(v,:));
            for u = N
                if Available(u)
                    Queue = [Queue u]; 
                    Available(u) = 0;
                end
            end
        end
        s(Explored==1) = sum(Explored);
        Explored= zeros(1,n);
    end
end