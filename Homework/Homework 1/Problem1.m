%% results for part (a)
% find the value a,b where a given k would give grad(f)=0
g = @(x) 1-cos(x);
ReLU = @(x) x.*(x>0);
for k = 0:5
    % construct linear eqns. Mx = y, x=(a,b)'
    X = (k:5) *pi/10; % vector of x_k,...,x_5

    M = [sum(X) -(5-k+1) ;
         X*X' -sum(X)];
    % rhs
    y = [sum(g(X));
        g(X)*(X')];

    x = (M^(-1)*y); % input (a', b')
    
    % solution to eq w/ fixed k
    fprintf("k' = %f, a' = %f, b' = %f\n", k,x(1),x(2))
    fprintf("resulting k(a',b') = %f\n", ceil(x(2)/x(1) *10/pi))
end

% results
figure % approximation over 0 to pi/2 of ReLU and g
a = 0.861289; 
b = 0.373492;
x = 0:.001:pi/2;
plot(x,g(x),x,ReLU(a*x-b))
legend('g(x)','approximation')

figure % loss function including this point
inputs = -.5:0.001:1.5;
[A,B] = meshgrid(inputs,inputs);
X = (0:5) * pi/10;
F = zeros(length(inputs));
f = @(a,b) 1/12*sum((ReLU(a*X-b)-g(X)).^2);
for i= 1:length(inputs)
    for j = 1:length(inputs)
        F(i,j) = f(A(i,j),B(i,j));
    end
end
mesh(A,B,F)
xlabel("a"), ylabel("b")
hold on
plot3(a,b,f(a,b),'r.','markersize',20)
zlim([0 0.25])
legend('loss function', 'analytic min')

% minimum of f
fprintf("\nf global min: %f\n", f(a,b))





%% Part (b) analytic computations
% find the gradient for part (b)
X = (0:5) * pi/10;
fa0 = (X-g(X))*X'/6;
fb0 = -sum(X-g(X))/6;
fprintf("initial gradient: gradf = (%f,%f)'\n",fa0,fb0)
% find alpha star
as = (pi/2)/(pi/2*fa0 - fb0);
fprintf("critical step size: %f\n",as)

g = @(x) 1-cos(x);
ReLU = @(x) x.*(x>0);
f = @(a,b) 1/12*sum((ReLU(a*X-b)-g(X)).^2);

% gradient descent
% gradient vector
N = 10000;
grad = @(a,b) [((a*X-b-g(X)).*(a*X-b > 0))*X' ;
    -sum((a*X-b-g(X)).*(a*X-b>=0))]/6;
F = zeros(1,N);
x = [1;0]; % initial a0,b0 as state vector
F(1) = f(x(1),x(2));
% naive gradient descent w/0 tolerence
% repeat varying constant before as for each figure...
alpha = 0.872*as;
for i = 2:N
    x = x - alpha*grad(x(1),x(2));
    F(i) = f(x(1),x(2));
end
figure
fmin = 0.000363;
plot(1:N,F,[1 N],[fmin fmin])
ylim([-0.05 .15])
legend('loss function', 'analytic min')
xlabel("iteration")

% find the best step-size
X = (k:5) *pi/10;
M = [X*X'/6 -sum(X)/6;
    -sum(X)/6 (5-k+1)/6];
eigs(M);
alpha_optimal = 2/(max(eigs(M))+min(eigs(M)));
fprintf("ideal step-size: %f\n", alpha_optimal)




%% part (c) stochastic gradient descent
N = 100;
grad1 = @(a,b) ((a*X-b-g(X)).*(a*X-b > 0))*X'/6;
grad2 = @(a,b) -sum((a*X-b-g(X)).*(a*X-b>=0))/6;
F = zeros(1,N);
x = [1;0]; % initial a0,b0 as state vector
F(1) = f(x(1),x(2));
% naive gradient descent w/0 tolerence
% repeat varying constant before as for each figure...
alpha = @(k) 1.5/k;
for i = 2:N
    index = randi(2); % whether to change a or b
    if index == 1
        x(index) = x(index) - alpha(i)*grad1(x(1),x(2));
    else
        x(index) = x(index) - alpha(i)*grad2(x(1),x(2));
    end
    F(i) = f(x(1),x(2));
end
figure
fmin = 0.000363;
plot(1:N,F,[1 N],[fmin fmin])
ylim([-0.05 .15])
legend('loss function', 'analytic min')
xlabel("iteration")

