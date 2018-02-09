function [result] = funcPerceptron(X, Y, W, MAX_ITERA, MAX_ITERASIZE, THRESHOLD)

% Init elements
N= size(X, 1);
DIM= size(X, 2);
result= zeros(1, DIM+1);

% Loop over iterations
for itera= 1:MAX_ITERA
    PERMU= randperm(N);
    for n= 1:min([N MAX_ITERASIZE])
%         PREDIC= sum(result.*[1 X(PERMU(n), :)]);
%         result= result + 0.01 * ((Y(PERMU(n))-PREDIC)) * N * W(PERMU(n)) .* [1 X(PERMU(n), :)];
        INDIC= sum(result.*[1 X(PERMU(n), :)]) > THRESHOLD;
        if (Y(PERMU(n))>0) ~= INDIC
            result= result + ((Y(PERMU(n))>0)-INDIC) * N * W(PERMU(n)) .* [1 0.01 0.01] .* [1 X(PERMU(n), :)];
        end
    end
end

end
