%% Prediction computation on test data
PREDIC= zeros(1, N);
ErrorCount= zeros(ADABOOST_ITERA+1, 1);
ErrorCount(1)= 0.5*N;
for t= 1:ADABOOST_ITERA
    if size(H, 2) == 4
        INDIC= ((X(:, 1)'-H(t, 1))*H(t, 3) + (X(:, 2)'-H(t, 2))*H(t, 4)) > 0;
    elseif size(H, 2) == 3
        INDIC= sum(repmat(H(t, :), N, 1).*[ones(N, 1) X], 2)' > THRESHOLD;
    end
    PREDIC= PREDIC + ALPHA(t)*(INDIC*2-1);
    ErrorCount(t+1)= sum((Y>0)~=(PREDIC'>0));
end


%% Field creation
fieldN= 100;
fieldM= 100;
MINX= min(min(X(:, 1)));
MINY= min(min(X(:, 2)));
MAXX= max(max(X(:, 1)));
MAXY= max(max(X(:, 2)));
FIELD= zeros(1, fieldN*fieldM);
FIELDX= zeros(fieldN*fieldM, 2);
FIELDDIM1= MINX+0.5*(MAXX-MINX)/fieldN:(MAXX-MINX)/fieldN:MAXX;
FIELDDIM2= MINY+0.5*(MAXY-MINY)/fieldM:(MAXY-MINY)/fieldM:MAXY;
for n= 1:fieldN
    for m= 1:fieldM
        FIELDX((n-1)*fieldM + m, 1)= n*(MAXX-MINX)/fieldN + MINX;
        FIELDX((n-1)*fieldM + m, 2)= m*(MAXY-MINY)/fieldM + MINY;
    end
end
for t= 1:ADABOOST_ITERA
    if size(H, 2) == 4
        INDIC= ((FIELDX(:, 1)'-H(t, 1))*H(t, 3) + (FIELDX(:, 2)'-H(t, 2))*H(t, 4)) > 0;
    elseif size(H, 2) == 3
        INDIC= (sum(repmat(H(t, :), fieldN*fieldM, 1).*[ones(fieldN*fieldM, 1) FIELDX], 2))' > THRESHOLD;
    end
    FIELD= FIELD + ALPHA(t)*(INDIC*2-1);
end
