function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

h1 = sigmoid([ones(m, 1) X] * Theta1');%[5000x401]x[401x25] = [5000x25]
h2 = sigmoid([ones(m, 1) h1] * Theta2');%[5000x26]x[26x10] = [5000x10]
%portanto cada linha(exemplo de treino) de h2 e um vetor linha de 10
%posicoes, que representa y=[0 0 1 ... 0] label=10.

[dummy, p] = max(h2, [], 2);
%p recebe um vetor com os indices dos maiores valores encontrados
%em cada linha de saida que o modelo calculou:
% index(max((h2(i)=[.....])))

%dummy recebe um vetor com os maiores valors encontrados em cada
%linha de saida que o modelo calculou:
% max((h2(i)=[.....]))

%M = max(A,[],dim) returns the largest elements along dimension dim. 
%For example, if A is a matrix, then max(A,[],2) is a column vector 
%containing the maximum value of each row.

% =========================================================================


end
