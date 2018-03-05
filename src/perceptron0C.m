% Perceptron 
% 1 ere couche 784 neurones - input
% 2 eme couche 10 neurones - output
% Warning : some of the short cut used work only with recent matlab version

clc
clear all
close all  

%% Setting Zone %%

images = loadMNISTImages('dasta/train-images.idx3-ubyte');
labels = loadMNISTLabels('data/train-labels.idx1-ubyte');

Nb_training   = 60000;
learning_rate = 0.001;
niter         = 1000;

training = randperm(length(labels),Nb_training);
lambda   = 1.;
sigmoid  = @(x) 1./(1+exp(-lambda*x));
dsigmoid = @(x) lambda*exp(-lambda*x)./(exp(-lambda*x) + 1).^2;

reference = zeros(10,1);
% Couches          % Poids              % Biais          % Intermédiaire  
a0 = zeros(784,1); 
a1 = zeros(10,1);  w1 = 2*rand(10,784) - 1;  b1 = 2*rand(10,1) - 1; z1 = zeros(10,1);
% dérivées partielles
dCdw1 = zeros(size(w1)); dCdb1 = zeros(length(b1));
% record energy 
cost = zeros(niter); % record mean energy after training over the set

%% Training Zone %%

for n = 1:niter
    for i = 1:Nb_training
        % Données en entrée
        a0 = images(:,training(i));

        % première couche
        z1 = w1*a0 + b1;
        a1 = sigmoid(z1);  

        reference = zeros(10,1);
        reference(labels(training(i))+1) = 1; 
        cost(n) = cost(n) + norm(reference - a1,2);
     
        % Backpropagation        
        dCdb1 = 2*(a1 - reference).*dsigmoid(z1);
        dCdw1 = ((2*ones(10,784).*(a1-reference)).*a0').*dsigmoid(z1);
        
        w1 = w1 - learning_rate*dCdw1; b1 = b1 - learning_rate*dCdb1;
    end   
    cost(n) = cost(n)/Nb_training;
    fprintf('Iteration : %3d cout : %f \n',n,cost(n));
end


%% Test Zone %%
tests       = loadMNISTImages('data/t10k-images.idx3-ubyte');
test_labels = loadMNISTLabels('data/t10k-labels.idx1-ubyte');

Nb_test = 10000; 
test = randperm(length(test_labels),Nb_test);
success = 0; devine = zeros(1,Nb_test);

for i = 1:Nb_test
   a0 = tests(:,test(i));
   guess = sigmoid(w1*a0 + b1);
   
   [~ , idx] = max(guess);
   devine(i) = idx - 1;
   if (idx - 1 == test_labels(test(i)))
       success = success + 1;
   end
end

fprintf('Efficacité : %f\n',success/Nb_test);
fprintf('Taux erreur : %f\n',1 - success/Nb_test);


%% Affichage
Nb_test = 20; 
test = randi([1 max(size(tests))],1,Nb_test);
success = 0; devine = zeros(1,Nb_test);

for i = 1:Nb_test
   a0 = tests(:,test(i));
   guess = sigmoid(w1*a0 + b1);
   
   [~ , idx] = max(guess);
   devine(i) = idx - 1;
   if (idx - 1 == test_labels(test(i)))
       success = success + 1;
   end
end

[devine ; test_labels(test)' ; logical(devine - test_labels(test)')]

for i= 1:Nb_test
    subplot(10, Nb_test, Nb_test*devine(i)+i);
    imagesc(reshape(tests(:,test(i)), 28, 28));
    axis off
    axis tight
end