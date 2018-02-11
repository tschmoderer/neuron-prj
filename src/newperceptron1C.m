clc
clear all
close all 


images = loadMNISTImages('data/train-images.idx3-ubyte');
labels = loadMNISTLabels('data/train-labels.idx1-ubyte');

Nb_training = 1000;
learning_rate = 0.05;
niter = 100;

training = randi([1 max(size(images))],1,Nb_training);

sigmoid  = @(x) 1./(1+exp(-x));
dsigmoid = @(x) exp(-x)./(exp(-x) + 1).^2;


E = 784; J = 16; S = 10;

yd = zeros(S,1); err = 1;

% Couches         % Poids             % Intermédiaire  
x = zeros(E,1);   w1 = rand(J,E + 1);     alphJ = zeros(J,1);  
z = zeros(J,1);   w2 = rand(S,J + 1);     alphS = zeros(S,1);

while err > 0.1
   for n = 1:Nb_training
       a1 = w1*[x ; 1];  x1 = tanh(a1);
       a2 = w2*[x1 ; 1]; y = tanh(a2);
       
       yd = zeros(S,1);
       yd(labels(training(n))+1) = 1; 
       err = norm(yd - y,2);
       
       errorS = -(yd-y).*(1-y.*y);
       Gradw2 = [x1 ; 1]*errorS';
       
       errorJ = (w2(:,1:end-1)'*errorS).*(1-x1.*x1);
       Gradw1 = [x ; 1]*errorJ';
       
       w1 = w1 - learning_rate*Gradw1';
       w2 = w2 - learning_rate*Gradw2';
       
%        % propage x --> z --> y
%        alphJ = w1*[x ; 1] ;
%        z = sigmoid(alphJ);
%        
%        alphS = w2*[z ; 1];
%        y = sigmoid(alphS);
%        
%        yd = zeros(S,1);
%        yd(labels(training(n))+1) = 1; 
%        err = norm(yd - y,2);
%        
%        % Calcul ErrorS 
%        errS = ones(S,J + 1);
% %        for s = 1:S
% %            for j = 1:J
% %                errS(s,j) = -(yd(s) - y(s))*dsigmoid(alphS(s))*z(j);
% %            end
% %        end
%        errS = -errS.*(yd - y).*dsigmoid(alphS).*z';
%        w2 = w2 - learning_rate*errS;
%        
%        % Calcul ErroJ
%        errJ = zeros(J,E);
%        for j = 1:J
% %            for e = 1:E
% %                somme = 0;
% %                for s = 1:S
% %                    somme = somme + (yd(s)-y(s))*dsigmoid(alphS(s))*w2(s,j)*x(e);
% %                end
% %                somme = sum((yd - y).*dsigmoid(alphS).*w2(:,j)*x(e));
% %                errJ(j,e) = -sum((yd - y).*dsigmoid(alphS).*w2(:,j)*x(e));
% %            end
%            errJ(j,:) = -sum((yd - y).*dsigmoid(alphS).*w2(:,j)).*x';
%        end
%        w1 = w1 - learning_rate*errJ;
   end
   err
end

tests       = loadMNISTImages('data/t10k-images.idx3-ubyte');
test_labels = loadMNISTLabels('data/t10k-labels.idx1-ubyte');

Nb_test = 50; 
test = randi([1 max(size(tests))],1,Nb_test);
success = 0; devine = zeros(1,Nb_test);

for i = 1:Nb_test
   x = tests(:,test(i));
   a1 = w1*[x ; 1];  x1 = tanh(a1);
   a2 = w2*[x1 ; 1]; y = a2;
   
   [~ , idx] = max(y);
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