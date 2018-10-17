function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
n = length(theta);
J_history = zeros(num_iters, 1);
lambda=1;


grad = 0;

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
    h = sigmoid(X*theta);
    grad=0;
    
    theta_ = theta;
    theta_(1)=0;
%Calculando grad Regulazirado Vetorialmente
    grad = grad + (1/m)*X'*(h-y) + (lambda/m)*eye(length(theta_))*theta_; %[(NxM) x (Mx1)] + [(NxN) x (Nx1)]= (Nx1)
        
    theta = theta -alpha*grad;

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = costJ(theta,X, y,lambda);

end

end
