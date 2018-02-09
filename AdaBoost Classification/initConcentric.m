%% Test data
Y= zeros(N, 1);
Y(1:N/4, 1)= -1;
Y(N/4+1:N, 1)= 1;

X= zeros(N, 2);
X(1:N/4, :)= randn(N/4, 2);
radii= 3+0.4*randn(3*N/4, 1);
theta= 2*pi*rand(3*N/4, 1);
X(N/4+1:N, 1)= radii.*cos(theta);
X(N/4+1:N, 2)= radii.*sin(theta);
