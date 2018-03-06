% Perceptron 
% 1 ere couche 784 neurones - input
% 2 eme couche 16 neurones
% 3 eme couche 16 neurones
% 4 eme couche 10 neurones - output

clc
clear all
close all  

%% Setting Zone %%

images      = loadMNISTImages('data/train-images.idx3-ubyte');
labels      = loadMNISTLabels('data/train-labels.idx1-ubyte');
tests       = loadMNISTImages('data/t10k-images.idx3-ubyte');
test_labels = loadMNISTLabels('data/t10k-labels.idx1-ubyte');

Nb_test       = 10000; 
Nb_training   = 60000;
learning_rate = 0.001;
niter         = 1000;

training = randperm(length(labels),Nb_training);
E = 784; C1 = 100; C2 = 16; S = 10;

lambda   = 1.;

f1  = @(x) 1./(1+exp(-lambda*x)); % sigmoide, activation couche 1
df1 = @(x) lambda*exp(-lambda*x)./(exp(-lambda*x) + 1).^2;

f2  = @(x) 1./(1+exp(-lambda*x)); % activation couche 2
df2 = @(x) lambda*exp(-lambda*x)./(exp(-lambda*x) + 1).^2;

fs  = @(x) x.*(x>0); % ReLU
dfs = @(x) x>0;

fs  = @(x) exp(x)/sum(exp(x(:))); %  activation couche S
dfs = @(x) sum((ones(S) - x.*ones(S))/sum(exp(x(:))))';

reference = zeros(S,1);

% Couches          % Poids            % Biais             % Intermédiaire  
a0 = zeros(E,1); 
a1 = zeros(C1,1);  w1 = 2*rand(C1,E) - 1;  b1 = 2*rand(C1,1) - 1; z1 = zeros(C1,1);
a2 = zeros(C2,1);  w2 = 2*rand(C2,C1) - 1; b2 = 2*rand(C2,1) - 1; z2 = zeros(C2,1);
a3 = zeros(S,1);   w3 = 2*rand(S,C2) - 1;  b3 = 2*rand(S,1) - 1;  z3 = zeros(S,1);

% dérivées partielles
dCdw1 = zeros([size(w1)]); dCdb1 = zeros([length(b1)]);
dCdw2 = zeros([size(w2)]); dCdb2 = zeros([length(b2)]);
dCdw3 = zeros([size(w3)]); dCdb3 = zeros([length(b3)]);
dCda1 = zeros(length(a1)); dCda2 = zeros(length(a2)); dCda3 = zeros(length(a3));
dCdz1 = zeros(length(z3)); dCdz2 = zeros(length(z2)); dCdz3 = zeros(length(z3));

% record energy 
cost = zeros(niter); % record mean energy after training over the set

for n = 1:niter
    for i = 1:Nb_training
        % Données en entrée
        a0 = images(:,training(i));

        % première couche cachée
        z1 = w1*a0 + b1;
        a1 = f1(z1);
        
        % deuxième couche cachée
        z2 = w2*a1 + b2;
        a2 = f2(z2);
        
        % Couche de sortie 
        z3 = w3*a2 + b3;
        a3 = fs(z3); 
        
        reference = zeros(10,1);
        reference(labels(training(i))+1) = 1; 
        cost(n) = cost(n) + norm(reference - a1,2);
        
        % Backpropagation  
        dCda3 = 2*(a3 - reference);
        dCdz3 = dfs(z3).*dCda3;
        
        dCdb3 = dCda3.*dfs(z3);
        dCdw3 = (ones(size(w3)).*a2').*dCdz3;
        
        dCda2 = sum(w3.*dCdz3)';
        dCdz2 = df2(z2).*dCda2;
        
        dCdb2 = dCda2.*df2(z2);
        dCdw2 = (ones(size(w2)).*a1').*dCdz2;
  
        dCda1 = sum(w2.*dCdz2)';
        dCdz1 = df1(z1).*dCda1;
        
        dCdb1 = dCda1.*df1(z1);
        dCdw1 = (ones(size(w1)).*a0').*dCdz1;      

        w1 = w1 - learning_rate*dCdw1; b1 = b1 - learning_rate*dCdb1;
        w2 = w2 - learning_rate*dCdw2; b2 = b2 - learning_rate*dCdb2;
        w3 = w3 - learning_rate*dCdw3; b3 = b3 - learning_rate*dCdb3;
    end
    cost(n) = cost(n)/Nb_training;
    if mod(n,10) == 0
       fprintf('Iteration : %3d, cout : %f, réussite : %f \n',n,mean(cost));
    end
end

%% Test Zone %%
test    = randperm(length(test_labels),Nb_test);
success = 0; devine = zeros(1,Nb_test);

for i = 1:Nb_test
   a0 = tests(:,test(i));
   guess = fs(w3*f2(w2*f1(w1*a0 + b1) + b2)+ b3);
   
   [~ , idx] = max(guess);
   devine(i) = idx - 1;
   if (idx - 1 == test_labels(test(i)))
       success = success + 1;
   end
end

fprintf('Efficacité : %f\n',success/Nb_test);
fprintf('Taux erreur : %f\n',1 - success/Nb_test);

%% Affichage %%
Nb_test = 20;

test = randi([1 max(size(tests))],1,Nb_test);
success = 0; devine = zeros(1,Nb_test);

for i = 1:Nb_test
   a0 = tests(:,test(i));
   guess = fs(w3*f2(w2*f1(w1*a0 + b1) + b2)+ b3);
   
   [~ , idx] = max(guess);
   devine(i) = idx - 1;
   if (idx - 1 == test_labels(test(i)))
       success = success + 1;
   end
end

[devine ; test_labels(test)' ; logical(devine -test_labels(test)') ]

for i= 1:Nb_test
    subplot(10, Nb_test, Nb_test*devine(i)+i);
    imagesc(reshape(tests(:,test(i)), 28, 28));
    axis off
    axis tight
end