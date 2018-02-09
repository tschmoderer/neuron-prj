%% Test data
Y= zeros(N, 1);
Y(1:N/3, 1)= -1;
Y(N/3+1:N, 1)= 1;

X= zeros(N, 2);
X(1:N/3, :)= [randn(N/3, 1) 2+randn(N/3, 1)];
X(N/3+1:2*N/3, :)= [4+randn(N/3, 1) 4+randn(N/3, 1)];
X(2*N/3+1:N, :)= [-8+randn(N/3, 1) 4+randn(N/3, 1)];
