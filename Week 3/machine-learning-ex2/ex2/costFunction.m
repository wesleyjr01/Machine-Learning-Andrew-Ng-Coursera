function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples
n = length(theta);

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
h_ = X*theta;
h = sigmoid(h_);

% Calculando J nao Regularizado, Vetorialmente
J = J + (1/m)*(-y'*log(h)-(1-y)'*log(1-h)); %J escalar
% end


% for k=1:n %%  SOLUCAO NAO VETORIAL
%     grad(k) = grad(k) + (1/m)*(h-y)'*X(:,k);
% end

%Calculando grad Vetorialmente
 grad = grad + (1/m)*X'*(h-y); %(3x100) x (100x1) = (3x1)

% =============================================================

end
