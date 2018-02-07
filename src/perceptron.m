% Perceptron 
% 1 ere couche 784 neurones - input
% 2 eme couche 16 neurones
% 3 eme couche 16 neurones 
% 4 eme couche 10 neurones - output

clc
% clear all
close all  

% images = loadMNISTImages('data/train-images.idx3-ubyte');
% labels = loadMNISTLabels('data/train-labels.idx1-ubyte');

Nb_training = 10;
batch_size  = 100;
niter = 1000;

training = randi([1 max(size(images))],1,Nb_training);
sigmoid  = @(x) 1./(1+exp(-x));
dsigmoid = @(x) exp(-x)./(exp(-x) + 1).^2;

reference = zeros(10,1);
% Couches          % Poids            % Biais          % Intermédiaire  
a0 = zeros(784,1); 
a1 = zeros(16,1);  w1 = rand(16,784); b1 = rand(16,1); z1 = zeros(16,1);
a2 = zeros(16,1);  w2 = rand(16,16);  b2 = rand(16,1); z2 = zeros(16,1);
a3 = zeros(10,1);  w3 = rand(10,16);  b3 = rand(10,1); z3 = zeros(10,1);

% dérivées partielles
dCdw3 = zeros([size(w3),Nb_training]); dCda3 = zeros([size(a3),Nb_training]); dCdb3 = zeros([size(b3),Nb_training]); 
dCdw2 = zeros([size(w2),Nb_training]); dCda2 = zeros([size(a2),Nb_training]); dCdb2 = zeros([size(b2),Nb_training]);
dCdw1 = zeros([size(w1),Nb_training]); dCda1 = zeros([size(a1),Nb_training]); dCdb1 = zeros([size(b1),Nb_training]);

for n = 1:niter
    cost = zeros(1,Nb_training);
    dCdw3 = zeros([size(w3),Nb_training]); dCda3 = zeros([size(a3),Nb_training]); dCdb3 = zeros([size(b3),Nb_training]); 
    dCdw2 = zeros([size(w2),Nb_training]); dCda2 = zeros([size(a2),Nb_training]); dCdb2 = zeros([size(b2),Nb_training]);
    dCdw1 = zeros([size(w1),Nb_training]); dCda1 = zeros([size(a1),Nb_training]); dCdb1 = zeros([size(b1),Nb_training]);

    for i = 1:Nb_training
        % Données en entrée
        a0 = images(:,training(i));

        % première couche cachée
        z1 = w1*a0 + b1;
        a1 = sigmoid(z1);  

        % deuxième couche cachée
        z2 = w2*a1 + b2;
        a2 = sigmoid(z2);

        % sortie
        z3 = w3*a2 + b3;
        a3 = sigmoid(z3);

        reference = zeros(10,1);
        reference(labels(training(i))+1) = 1; 
        cost(i) = norm(reference - a3,2);
     
% Backpropagation         
        dCda3(:,:,i) = 2*(a3 - reference);         % je suis sur
        dCdb3(:,:,i) = dsigmoid(z3).*dCda3(:,:,i); % je suis sur
        for j = 1:size(w3,1)     % i.e. 10
            for k = 1:size(w3,2) % i.e. 16
                dCdw3(j,k,i) = a2(k)*dsigmoid(z3(j))*dCda3(j,:,i); % je suis sur
            end
        end

        % deuxième couche
        for m = 1:16
            for j = 1:10
                dCda2(m,:,i) = dCda2(m,:,i) + w3(j,m)*dsigmoid(z3(j))*dCda3(j,:,i);
                dCdb2(m,:,i) = dCdb2(m,:,i) + dsigmoid(z3(j))*dCda3(j,:,i);
            end
        end
        
        for j = 1:16
            for k = 1:16
            dCdw2(j,k,i) = a1(j)*dsigmoid(z2(k))*dCda2(k,:,i);
            end
        end
        
        % première couche
        for m = 1:16
            for j = 1:16
                dCda1(m,:,i) = dCda1(m,:,i) + w2(j,m)*dsigmoid(z2(j))*dCda2(j,:,i);
                dCdb1(m,:,i) = dCdb1(m,:,i) + dsigmoid(z2(j))*dCda2(j,:,i);
            end
        end
        
        for j = 1:784
            for k = 1:16
            dCdw1(k,j,i) = a0(j)*dsigmoid(z1(k))*dCda1(k,:,i);
            end
        end            
    end
    
    w3 = w3 - mean(dCdw3,3); b3 = b3 - mean(dCdb3,3);
    w2 = w2 - mean(dCdw2,3); b2 = b2 - mean(dCdb2,3);
    w1 = w1 - mean(dCdw1,3); b1 = b1 - mean(dCdb1,3);
    
    [n mean(cost)]
end

% tests       = loadMNISTImages('data/t10k-images.idx3-ubyte');
% test_labels = loadMNISTLabels('data/t10k-labels.idx1-ubyte');

Nb_test = 1000; 
test = randi([1 max(size(tests))],1,Nb_test);
success = 0;

for i = 1:Nb_test
   a0 = tests(:,test(i));
   guess = sigmoid(w3*sigmoid(w2*sigmoid(w1*a0 + b1) + b2) + b3);
   
   [m , idx] = max(guess);
   if (idx-1 == test_labels(test(i)))
       success = success + 1;
   end
end

fprintf('Efficacité : %f\n',success/Nb_test);
fprintf('Taux erreur : %f\n',1 - success/Nb_test);
