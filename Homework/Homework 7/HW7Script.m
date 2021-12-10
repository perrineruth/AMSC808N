%% Problem 1.c find component ratios
G0 = @(z,p) p*z + (1-p)*z.^3;
km = @(p) 3-2*p;
H1 = @(z,p) ( km(p) - sqrt(km(p)^2-12*z.^2*p*(1-p)) ) ./ (6*(1-p)*z);
H0 = @(z,p) z.*G0(H1(z,p),p);

p = .125;
Ps = zeros(1,100);
for i = 1:100
    Ps(i) = GFInt(@(z) H0(z,p),i,100);
end
plot(Ps)