close all

%% Algorithm matrices
HDIM= 4;
W= ones(ADABOOST_ITERA+1, N)/N;
ALPHA= zeros(ADABOOST_ITERA, 1);
H= zeros(ADABOOST_ITERA, HDIM);


%% Learning iterative loop
for t= 1:ADABOOST_ITERA
    H(t, :)= funcParallelSep(X, Y, W(t, :), PARALLELSEP_STEP);
    INDIC= ((X(:, 1)-H(t, 1))*H(t, 3) + (X(:, 2)-H(t, 2))*H(t, 4)) > 0;
    Epsilon= sum(W(t, :)'.*((INDIC*2-1) ~= Y));
    ALPHA(t)= 0.5*log((1-Epsilon)/Epsilon);
    Z= sum(W(t, :)'.*exp(-Y.*ALPHA(t).*(INDIC*2-1)));
    W(t+1, :)= (W(t, :)'.*exp(-Y.*ALPHA(t).*(INDIC.*2-1)))/Z;
end

clear t
clear Z
clear W
clear HDIM
clear INDIC
clear Epsilon


%% Result Computation
run('ResultCompute.m')


%% Result display
run('ResultDisplay.m')
