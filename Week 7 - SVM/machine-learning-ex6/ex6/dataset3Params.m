function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C_ = [0.01 0.03 0.1 0.3 1 3 10 30];
sigma_ = C_;
Jval = zeros(length(C_),length(sigma_)); %Matriz de valores
Jval_ = zeros(length(C_),1);

%para cada linha, um vetor de valores com modelo(sigma_(j))
C = 0;
sigma = 0;
% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
for i=1:length(C_)
    for j=1:length(sigma_)
        model= svmTrain(X, y, C_(i), @(x1, x2) gaussianKernel(x1, x2, sigma_(j)));
        visualizeBoundary(X, y, model);
        y_pred = svmPredict(model,Xval); %vetor de previsoes para o dataset de validacao
        Jval(i,j) = mean(double(y_pred ~= yval));
    end
end
%%
[Row Colum] = MinIndex(Jval);

C = C_(Row);
sigma = sigma_(Colum);

% =========================================================================

end
