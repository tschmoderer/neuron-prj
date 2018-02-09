%% Result display
figure
hold on
axis([MINX MAXX MINY MAXY])
title('Ground truth')
plot(X(Y<0, 1), X(Y<0, 2), 'r.')
plot(X(Y>0, 1), X(Y>0, 2), 'b.')
axis equal

figure
hold on
axis([MINX MAXX MINY MAXY])
title('Classification result')
colormap jet
contour(FIELDDIM1, FIELDDIM2, reshape(-FIELD, fieldN, fieldM), [0 0], 'LineColor', [0 1 0], 'LineWidth', 2);
plot(X(PREDIC<0, 1), X(PREDIC<0, 2), 'r.')
plot(X(PREDIC>0, 1), X(PREDIC>0, 2), 'b.')
axis equal

figure
hold on
colormap jet
axis([MINX MAXX MINY MAXY])
title('Activation field')
surf(FIELDDIM1, FIELDDIM2, reshape(-FIELD, fieldN, fieldM),'EdgeColor','none')

figure
hold on
axis([0 ADABOOST_ITERA 0 1])
title('Error Rate')
plot(0:ADABOOST_ITERA, ErrorCount/N, '-')
