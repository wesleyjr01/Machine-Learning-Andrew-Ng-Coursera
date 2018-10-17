function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples
n = length(theta);
% You need to return the following variables correctly 
J = 0;
regul = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
h_ = X*theta;
h = sigmoid(h_);


regul = regul + theta'*theta;


% Calculando J Regularizado, Vetorialmente
J = J + (1/m)*(-y'*log(h)-(1-y)'*log(1-h)) + (lambda/(2*m))*regul; %J escalar
% end


% for k=1:n
%     if k==1
%         grad(k) = grad(k) + (1/m)*(h-y)'*X(:,k);
%     else
%         grad(k) = grad(k) + (1/m)*(h-y)'*X(:,k) + (lambda/m)*theta(k);
%     end
% end

theta_ = theta;
theta_(1)=0;
%Calculando grad Vetorialmente
 grad = grad + (1/m)*X'*(h-y) + (lambda/m)*eye(length(theta_))*theta_; %[(NxM) x (Mx1)] + [(NxN) x (Nx1)]= (Nx1)

% =============================================================

% =============================================================

end
