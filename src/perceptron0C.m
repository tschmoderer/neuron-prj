% Perceptron 
% 1 ere couche 784 neurones - input
% 2 eme couche 10 neurones - output

clc
clear all
close all  

images = loadMNISTImages('data/train-images.idx3-ubyte');
labels = loadMNISTLabels('data/train-labels.idx1-ubyte');

Nb_training = 500;
learning_rate = 0.1;
batch_size  = 100;
niter = 100;

training = randi([1 max(size(images))],1,Nb_training);
lambda   = 1.;
sigmoid  = @(x) 1./(1+exp(-lambda*x));
dsigmoid = @(x) lambda*exp(-lambda*x)./(exp(-lambda*x) + 1).^2;

reference = zeros(10,1);
% Couches          % Poids              % Biais          % Intermédiaire  
a0 = zeros(784,1); 
a1 = zeros(10,1);  w1 = zeros(10,784);  b1 = zeros(10,1); z1 = zeros(10,1);

for n = 1:niter
    cost = zeros(1,Nb_training);
    % dérivées partielles
    dCdw1 = zeros([size(w1),Nb_training]); dCdb1 = zeros([length(b1),Nb_training]);

    for i = 1:Nb_training
        % Données en entrée
        a0 = images(:,training(i));

        % première couche cachée
        z1 = w1*a0 + b1;
        a1 = sigmoid(z1);  

        reference = zeros(10,1);
        reference(labels(training(i))+1) = 1; 
        cost(i) = norm(reference - a1,2);
     
        % Backpropagation        
        dCdb1(:,i)   = 2*(a1 - reference).*dsigmoid(z1);
        dCdw1(:,:,i) = ((2*ones(10,784).*(a1-reference)).*a0').*dsigmoid(z1);
    end
    
     w1 = w1 - learning_rate*mean(dCdw1,3); b1 = b1 - learning_rate*mean(dCdb1,2);
     
     [n mean(cost)]
end

tests       = loadMNISTImages('data/t10k-images.idx3-ubyte');
test_labels = loadMNISTLabels('data/t10k-labels.idx1-ubyte');

Nb_test = 50; 
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

fprintf('Efficacité : %f\n',success/Nb_test);
fprintf('Taux erreur : %f\n',1 - success/Nb_test);

[devine ; test_labels(test)' ; logical(devine -test_labels(test)') ]

for i= 1:Nb_test
    subplot(10, Nb_test, Nb_test*devine(i)+i);
    imagesc(reshape(tests(:,test(i)), 28, 28));
    axis off
    axis tight
end