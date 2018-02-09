%% Test data
Y= zeros(N, 1);
X= zeros(N, 2);

Spreadness= 0.25;
Pattern= [
    [2 2 0 0 1];
    [2 1 1 2 0];
    [0 1 2 2 0];
    [2 1 1 2 0];
    [2 2 0 0 1];
    ];

Pattern= fliplr(Pattern');

nbSpots= sum(sum(Pattern~=0));
numSpot= 1;
iStart= 1;
for n= 1:size(Pattern, 1)
    for m= 1:size(Pattern, 1)
        if Pattern(n, m) ~= 0
            iEnd= round((numSpot==nbSpots)*N + (numSpot~=nbSpots)*(iStart+N/nbSpots));
            Y(iStart:iEnd, 1)= (Pattern(n, m) == 1)*2-1;
            X(iStart:iEnd, :)= [n+Spreadness*randn(iEnd-iStart+1, 1) m+Spreadness*randn(iEnd-iStart+1, 1)];
            numSpot= numSpot+1;
            iStart= iEnd+1;
        end
    end
end

clear Spreadness
clear Pattern
clear nbSpots
clear numSpot
clear nbDots
clear iStart
clear iEnd
clear n
clear m
