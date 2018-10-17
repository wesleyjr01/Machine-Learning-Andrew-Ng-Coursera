function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples
n = length(theta);

% You need to return the following variables correctly 
J = 0;
regul = 0;
grad = zeros(size(theta));
theta_ = theta;
theta_(1)=0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Hint: The computation of the cost function and gradients can be
%       efficiently vectorized. For example, consider the computation
%
%           sigmoid(X * theta)
%
%       Each row of the resulting matrix will contain the value of the
%       prediction for that example. You can make use of this to vectorize
%       the cost function and gradient computations. 
%
% Hint: When computing the gradient of the regularized cost function, 
%       there're many possible vectorized solutions, but one solution
%       looks like:
%           grad = (unregularized gradient for logistic regression)
%           temp = theta; 
%           temp(1) = 0;   % because we don't add anything for j = 0  
%           grad = grad + YOUR_CODE_HERE (using the temp variable)
%

% X = [ones(m,1),X];
h_ = X*theta;
h = sigmoid(h_);
regul = regul + theta_'*theta_;

% Calculando J nao Regularizado, Vetorialmente
J = J + (1/m)*((-y'*log(h)-(1-y)'*log(1-h))) + (lambda/(2*m))*regul; %J Regularizado escalar 
% J = J + (1/m)*sum((-y.*log(h)-(1-y).*log(1-h))); %J nao-Regularizado escalar 
% end


% for k=1:n %%  SOLUCAO NAO VETORIAL
%     grad(k) = grad(k) + (1/m)*(h-y)'*X(:,k);
% end

%Calculando grad Vetorialmente

%Calculando grad Vetorialmente
 grad = grad + (1/m)*X'*(h-y) + (lambda/m)*eye(length(theta_))*theta_; %[(NxM) x (Mx1)] + [(NxN) x (Nx1)]= (Nx1)



% =============================================================

grad = grad(:);

end
