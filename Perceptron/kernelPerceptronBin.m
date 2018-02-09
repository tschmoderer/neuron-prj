%% Cleanup
clearvars -except RAW_TRAIN_DATA RAW_TEST_DATA
close all
close all hidden


%% Init parameters
NB_TRAIN_IMAGES= 1000;
NB_TEST_IMAGES= 1000;
DISPLAY= 0;


%% Init training and test sets
TRAIN_IMAGES= RAW_TRAIN_DATA(1:NB_TRAIN_IMAGES, :);
TEST_IMAGES= RAW_TEST_DATA(1:NB_TEST_IMAGES, :);


%% Binarize the classes
DIGIT_TO_RECOGNIZE= 1;
TRAIN_IMAGES(:, end)= (TRAIN_IMAGES(:, end)==DIGIT_TO_RECOGNIZE)*2-1;
TEST_IMAGES(:, end)= (TEST_IMAGES(:, end)==DIGIT_TO_RECOGNIZE)*2-1;


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
a= zeros(NB_TRAIN_IMAGES, 1);
for t= 1:MAX_ITERA
    converged= 1;
    for i= 1:NB_TRAIN_IMAGES
        predic= sum(a.*TRAIN_IMAGES(:, end).*KERNEL_MATRIX(:, i));
        if (predic>=0 && TRAIN_IMAGES(i, end)<0) || (predic<=0 && TRAIN_IMAGES(i, end)>0)
            a(i)= a(i)+1;
            converged= 0;
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
    predic=0;
    for j=1:NB_TRAIN_IMAGES
        KERNEL= exp(-norm(TEST_IMAGES(i, 1:end-1)-TRAIN_IMAGES(j, 1:end-1), 1)/(2*SIGMA^2));
        predic= predic+a(j)*TRAIN_IMAGES(j, end)*KERNEL;
    end
    
    if (predic>0 && TEST_IMAGES(i, end)>0) || (predic<=0 && TEST_IMAGES(i, end)<=0)
        ratio= ratio+1;
    end
    
    if DISPLAY==1
        subplot(2, NB_TEST_IMAGES, i+(predic<0)*NB_TEST_IMAGES);
        imagesc(reshape(TEST_IMAGES(i, 1:end-1), 16, 16)');
        axis off
        axis tight
    end
end

if DISPLAY==1
    tightfig;
end

ratio= ratio/NB_TEST_IMAGES
