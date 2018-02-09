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


%% Training
ITERA= 10;
w= zeros(Q, size(TRAIN_IMAGES(1, :), 2)-1);
for t= 1:ITERA
    for i= 1:NB_TRAIN_IMAGES
        [~, predic]= max(w*TRAIN_IMAGES(i, 1:end-1)');
        predic= predic-1;
        if predic ~= TRAIN_IMAGES(i, end)
            w(predic+1, :)= w(predic+1, :) - TRAIN_IMAGES(i, 1:end-1);
            w(TRAIN_IMAGES(i, end)+1, :)= w(TRAIN_IMAGES(i, end)+1, :) + TRAIN_IMAGES(i, 1:end-1);
        end;
    end;
end;


%% Testing
ratio= 0;
confuMat= zeros(Q, Q);
for i= 1:NB_TEST_IMAGES
    [~, predic]= max(w*TEST_IMAGES(i, 1:end-1)');
    predic= predic-1;
    if predic == TEST_IMAGES(i, end)
        ratio= ratio+1;
    end;
    
    confuMat(TEST_IMAGES(i, end)+1, predic+1)= confuMat(TEST_IMAGES(i, end)+1, predic+1)+1;
    if DISPLAY==1
        subplot(Q, NB_TEST_IMAGES, NB_TEST_IMAGES*predic+i);
        imagesc(reshape(TEST_IMAGES(i, 1:end-1), 16, 16)');
        axis tight
        axis off
    end;
end;

if DISPLAY==1
    tightfig;
end;

confuMat= confuMat./NB_TEST_IMAGES

ratio= ratio/NB_TEST_IMAGES
