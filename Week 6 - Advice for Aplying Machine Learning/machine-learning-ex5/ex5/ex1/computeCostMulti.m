function J = computeCostMulti(X, y, theta)
%COMPUTECOSTMULTI Compute cost for linear regression with multiple variables
%   J = COMPUTECOSTMULTI(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples
J_ = 0;
J = 0;

J = (1/(2*m))*((X*theta - y).')*(X*theta - y);% [Mx(N+1])x[(N+1)x1]x[Mx(N+1])x[(N+1)x1]' = escalar

% =========================================================================

end
