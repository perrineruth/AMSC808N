function p = GFInt(G,k,n)
% GFInt: Integrates a Generating Function around the unit circle in
%   the complex plane to gather a probably value
%   - G = Generating Function
%   - k = term in series
%   - n = number points

% change to an int in theta where dz = i*e^(i*th) dth

% composite trapezoid rule is just a regular Riemann sum in a
% closed loop... Don't do endpoints twice
th = linspace(0,2*pi,n+1); % points
th(end) = []; % endpoints already covered :)

% points, step-size
z = exp(1i*th);
dth = 2*pi/n;
% z = e^(i*th) over unit circle
p = dth * sum(G(z)./z.^(k)*1i)/(2*pi*1i);
if abs(imag(p)) > 1e-4
    error('large imaginary value')
end
p = real(p);
end