%% Cleanup
clearvars -except RAW_TRAIN_DATA RAW_TEST_DATA
close all
close all hidden


%% Init parameters
NB_TRAIN_IMAGES= 1000;
NB_TEST_IMAGES= 30;
DISPLAY= 1;


%% Init training and test sets
TRAIN_IMAGES= RAW_TRAIN_DATA(1:NB_TRAIN_IMAGES, :);
TEST_IMAGES= RAW_TEST_DATA(1:NB_TEST_IMAGES, :);


%% Binarize the classes
DIGIT_TO_RECOGNIZE= 0;
TRAIN_IMAGES(:, end)= (TRAIN_IMAGES(:, end)==DIGIT_TO_RECOGNIZE)*2-1;
TEST_IMAGES(:, end)= (TEST_IMAGES(:, end)==DIGIT_TO_RECOGNIZE)*2-1;


%% Training
ITERA= 10;
w= zeros(1, size(TRAIN_IMAGES(1, :), 2)-1);
for t= 1:ITERA
    for i= 1:NB_TRAIN_IMAGES
        if TRAIN_IMAGES(i, end)*(w*TRAIN_IMAGES(i, 1:end-1)') <= 0
            w= w + TRAIN_IMAGES(i, end)*TRAIN_IMAGES(i, 1:end-1);
        end;
    end;
end;


%% Testing
confuMat= zeros(2, 2);
for i= 1:NB_TEST_IMAGES
    predic= w*TEST_IMAGES(i, 1:end-1)';
    
    confuMat((TEST_IMAGES(i, end)>0)+1, (predic>0)+1)= confuMat((TEST_IMAGES(i, end)>0)+1, (predic>0)+1)+1;
    
    if DISPLAY==1
        subplot(2, NB_TEST_IMAGES, i+(predic<0)*NB_TEST_IMAGES);
        imagesc(reshape(TEST_IMAGES(i, 1:end-1), 16, 16)');
        axis tight
        axis off
    end;
end;

if DISPLAY==1
    tightfig;
end;

confuMat= confuMat./NB_TEST_IMAGES

ratio= confuMat(1,1) + confuMat(2,2)
