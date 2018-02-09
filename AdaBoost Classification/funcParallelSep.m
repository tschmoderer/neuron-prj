function [result] = funcParallelSep(X, Y, d, STEP)
    bestCost= zeros(5, 1);
    minx= min(X(:, 1));
    maxx= max(X(:, 1));
    miny= min(X(:, 2));
    maxy= max(X(:, 2));

    bestCost(1)= sum(X(Y<0, 1)>minx) + sum(X(Y>0, 1)<minx);

    for i=1:STEP
        %       |
        % moins | plus
        %       |
        v= minx + i*(maxx-minx)/STEP;
        cost= sum((X(:, 1)>v).*d'.*(Y<0)) + sum((X(:, 1)<v).*d'.*(Y>0));
        if cost < bestCost(1)
            bestCost(1)= cost;
            bestCost(2)= v; bestCost(3)= 0;
            bestCost(4)= 1; bestCost(5)= 0;
        end
        
        %      |
        % plus | moins
        %      |
        v= minx + i*(maxx-minx)/STEP;
        cost= sum((X(:, 1)>v).*d'.*(Y>0)) + sum((X(:, 1)<v).*d'.*(Y<0));
        if cost < bestCost(1)
            bestCost(1)= cost;
            bestCost(2)= v; bestCost(3)= 0;
            bestCost(4)= -1; bestCost(5)= 0;
        end
        
        % plus
        % -----
        % moins
        v= miny + i*(maxy-miny)/STEP;
        cost= sum((X(:, 2)>v).*d'.*(Y<0)) + sum((X(:, 2)<v).*d'.*(Y>0));
        if cost < bestCost(1)
            bestCost(1)= cost;
            bestCost(2)= 0; bestCost(3)= v;
            bestCost(4)= 0; bestCost(5)= 1;
        end

        % moins
        % -----
        % plus
        v= miny + i*(maxy-miny)/STEP;
        cost= sum((X(:, 2)>v).*d'.*(Y>0)) + sum((X(:, 2)<v).*d'.*(Y<0));
        if cost < bestCost(1)
            bestCost(1)= cost;
            bestCost(2)= 0; bestCost(3)= v;
            bestCost(4)= 0; bestCost(5)= -1;
        end
    end
    result= bestCost(2:5);
end

