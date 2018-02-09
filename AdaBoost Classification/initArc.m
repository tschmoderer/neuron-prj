%% Test data
Y= zeros(N, 1);
Y(1:N/3, 1)= -1;
Y(N/3+1:N, 1)= 1;

X= zeros(N, 2);
X(1:N/3, :)= 0.8*randn(N/3, 2);
radii= 3+0.5*randn(2*N/3, 1);
theta= pi/3 + pi*rand(2*N/3, 1) + 0.1*randn(2*N/3, 1);
X(N/3+1:N, 1)= radii.*cos(theta);
X(N/3+1:N, 2)= radii.*sin(theta);
