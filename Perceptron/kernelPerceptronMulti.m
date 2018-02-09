%% Cleanup
clearvars -except RAW_TRAIN_DATA RAW_TEST_DATA
close all
close all hidden


%% Init parameters
NB_TRAIN_IMAGES= 1000;
NB_TEST_IMAGES= 30;
DISPLAY= 1;
Q= 10;


%% Init training and test sets
TRAIN_IMAGES= RAW_TRAIN_DATA(1:NB_TRAIN_IMAGES, :);
TEST_IMAGES= RAW_TEST_DATA(1:NB_TEST_IMAGES, :);


%% Precomputations
SIGMA= 5;
KERNEL_MATRIX= zeros(NB_TRAIN_IMAGES, NB_TRAIN_IMAGES);
for i= 1:NB_TRAIN_IMAGES
    for j= 1:NB_TRAIN_IMAGES
        KERNEL_MATRIX(i, j)= exp(-norm(TRAIN_IMAGES(j, 1:end-1)-TRAIN_IMAGES(i, 1:end-1), 1)/(2*SIGMA^2));
    end
end
% imagesc(KERNEL_MATRIX-eye(NB_TRAIN_IMAGES, NB_TRAIN_IMAGES))


%% Training
MAX_ITERA= 10;
a= zeros(NB_TRAIN_IMAGES, Q);
for t= 1:MAX_ITERA
    converged= 1;
    for i= 1:NB_TRAIN_IMAGES
        for k=1:Q
            predic= sum(a(:, k).*((TRAIN_IMAGES(:, end)==k-1)*2-1).*KERNEL_MATRIX(:, i));
            if (predic>=0 && ((TRAIN_IMAGES(i, end)==k-1)*2-1)<0) || (predic<=0 && ((TRAIN_IMAGES(i, end)==k-1)*2-1)>0)
                a(i, k)= a(i, k)+1;
                converged= 0;
            end
        end
    end
    if converged == 1
        break
    end
end


%% TODO optimize by filtering the subset of selected training examples


%% Testing
ratio= 0;
for i= 1:NB_TEST_IMAGES
    predic= zeros(Q, 1);
    for j= 1:NB_TRAIN_IMAGES
        KERNEL= exp(-norm(TEST_IMAGES(i, 1:end-1)-TRAIN_IMAGES(j, 1:end-1), 1)/(2*SIGMA^2));
        for k=1:Q
            predic(k)= predic(k)+a(j, k)*((TRAIN_IMAGES(j, end)==k-1)*2-1)*KERNEL;
        end
    end
    [~, finalPredic]= max(predic);
    finalPredic= finalPredic-1;
    if (finalPredic == TEST_IMAGES(i, end))
        ratio= ratio+1;
    end
    if DISPLAY==1
        subplot(Q, NB_TEST_IMAGES, NB_TEST_IMAGES*finalPredic+i);
        imagesc(reshape(TEST_IMAGES(i, 1:end-1), 16, 16)');
        axis off
        axis tight
    end
end

if DISPLAY==1
    tightfig;
end

ratio= ratio/NB_TEST_IMAGES
