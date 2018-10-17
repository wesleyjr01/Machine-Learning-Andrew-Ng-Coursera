function sim = gaussianKernel(x1, x2, sigma)
%RBFKERNEL returns a radial basis function kernel between x1 and x2
%   sim = gaussianKernel(x1, x2) returns a gaussian kernel between x1 and x2
%   and returns the value in sim

% Ensure that x1 and x2 are column vectors
%x1 e um vetor de features [Mx1]
%x2 e um vetor de features [Mx1]
x1 = x1(:); x2 = x2(:);
%x1[Mx1] , x2[Mx1]
%f[Mx1]
m=length(x1);
f=zeros(m,1);

% You need to return the following variables correctly.
sim = 0;
norm=0;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the similarity between x1
%               and x2 computed using a Gaussian kernel with bandwidth
%               sigma
%
for i=1:m
    norm = norm + ((x1(i)-x2(i))^2)/(2*sigma^2);
end
sim = exp(-norm);





% =============================================================
    
end
