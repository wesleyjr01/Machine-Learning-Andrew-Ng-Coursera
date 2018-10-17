function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples
n = length(theta);

% You need to return the following variables correctly 
J = 0;
regul = 0;
grad = zeros(size(theta));

h = X*theta;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%
theta_ = theta;
theta_(1)=0;
regul = theta_'*theta;

J = (1/(2*m))*((X*theta - y).')*(X*theta - y) + (lambda/(2*m))*regul;

for k=1:n
    if k==1
        grad(k) = grad(k) + (1/m)*(h-y)'*X(:,k);
    else
        grad(k) = grad(k) + (1/m)*(h-y)'*X(:,k) + (lambda/m)*theta(k);
    end
end


% =========================================================================

grad = grad(:);

end
