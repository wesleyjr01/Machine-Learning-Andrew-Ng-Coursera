function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);
v = zeros(num_labels,1);

%Add a colum of zeros to X
X = [ones(m,1) X];



% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%
for i=1:m %para cada caso de treinamento
    
    z2 = Theta1*X(i,:)'; %calcula z(k+1)=Theta(k)*a(k) para o i-esimo exemplo, (25x401)x(401x1)
    a2 = sigmoid(z2); %a(k+1) = sigmoid(z(k+1)), k+1 e a camada que recebe esses valores, (25x1)  
    a2 = [1;a2]; %adiciona o bias nesta camada (26x1)
            
    z3 = Theta2*a2; %(10x26)x(26x1)  
    a3 = sigmoid(z3); %camada de saida (10x1)
    
    p(i) = find(a3 == max(a3(:)));% a i-esima posicao do vetor p(Mx1) recebe o maior valor da saida(a3)
        
end



% =========================================================================


end
