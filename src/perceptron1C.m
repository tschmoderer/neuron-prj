% Perceptron 
% 1 ere couche 784 neurones - input
% 2 eme couche 16 neurones
% 3 eme couche 10 neurones - output

clc
clear all
close all  

images = loadMNISTImages('data/train-images.idx3-ubyte');
labels = loadMNISTLabels('data/train-labels.idx1-ubyte');

Nb_training = 60000;
learning_rate = 0.001;
batch_size  = 100;
niter = 2000;

training = randi([1 max(size(images))],1,Nb_training); training = [1:Nb_training];
lambda   = 1.;
sigmoid  = @(x) 1./(1+exp(-lambda*x));
dsigmoid = @(x) lambda*exp(-lambda*x)./(exp(-lambda*x) + 1).^2;

reference = zeros(10,1);
% Couches          % Poids            % Biais             % Intermédiaire  
a0 = zeros(784,1); 
a1 = zeros(16,1);  w1 = 2*rand(16,784) - 1;  b1 = 2*rand(16,1) - 1; z1 = zeros(16,1);
a2 = zeros(10,1);  w2 = 2*rand(10,16) - 1;   b2 = 2*rand(10,1) - 1; z2 = zeros(10,1);

for n = 1:niter
    cost = zeros(1,Nb_training);
    % dérivées partielles
    dCdw1 = zeros(size(w1)); dCdb1 = zeros(length(b1));
    dCdw2 = zeros(size(w2)); dCdb2 = zeros(length(b2));
    
    dCda1 = zeros(length(a1)); dCda2 = zeros(length(a2));
    dCdz1 = zeros(length(z2)); dCdz2 = zeros(length(z2));
    
    for i = 1:Nb_training
        % Données en entrée
        a0 = images(:,training(i));

        % première couche cachée
        z1 = w1*a0 + b1;
        a1 = sigmoid(z1);
        
        % Couche de sortie 
        z2 = w2*a1 + b2;
        a2 = sigmoid(z2); 
        
        reference = zeros(10,1);
        reference(labels(training(i))+1) = 1; 
        cost(i) = norm(reference - a2,2);
     
        % Backpropagation  
        dCda2        = 2*(a2 - reference);
        dCdz2        = dsigmoid(z2).*dCda2;
        
        dCdb2        = dCda2.*dsigmoid(z2);
        dCdw2        = (ones(size(w2)).*a1').*dCdz2;
  
        dCda1        = sum(w2.*dCdz2)';
        dCdz1        = dsigmoid(z1).*dCda1;
        
        dCdb1        = dCda1.*dsigmoid(z1);
        dCdw1        = (ones(size(w1)).*a0').*dCdz1;      

        w1 = w1 - learning_rate*dCdw1; b1 = b1 - learning_rate*dCdb1;
        w2 = w2 - learning_rate*dCdw2; b2 = b2 - learning_rate*dCdb2;
    end
        
     fprintf('Iteration : %d cout : %f \n',n,mean(cost));
end

tests       = loadMNISTImages('data/t10k-images.idx3-ubyte');
test_labels = loadMNISTLabels('data/t10k-labels.idx1-ubyte');

Nb_test = 10000; 
test = randi([1 max(size(tests))],1,Nb_test); test = [1:Nb_test];
success = 0; devine = zeros(1,Nb_test);

for i = 1:Nb_test
   a0 = tests(:,test(i));
   guess = sigmoid(w2*sigmoid(w1*a0 + b1) + b2);
   
   [~ , idx] = max(guess);
   devine(i) = idx - 1;
   if (idx - 1 == test_labels(test(i)))
       success = success + 1;
   end
end

fprintf('Efficacité : %f\n',success/Nb_test);
fprintf('Taux erreur : %f\n',1 - success/Nb_test);

[devine ; test_labels(test)' ; logical(devine -test_labels(test)') ]

for i= 1:Nb_test
    subplot(10, Nb_test, Nb_test*devine(i)+i);
    imagesc(reshape(tests(:,test(i)), 28, 28));
    axis off
    axis tight
end