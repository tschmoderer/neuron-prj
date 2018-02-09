%% Test data
Y= zeros(N, 1);
Y(1:N/2, 1)= -1;
Y(N/2+1:N, 1)= 1;

X= zeros(N, 2);
X(1:N/2, :)= [randn(N/2, 1) 2+randn(N/2, 1)];
X(N/2+1:N, :)= [4+randn(N/2, 1) 4+randn(N/2, 1)];
