function X = featureDenormalize(X_norm, mu, sigma)
%FEATURENORMALIZE Normalizes the features in X 
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. This is often a good preprocessing step to do when
%   working with learning algorithms.

% You need to set these values correctly
X_ = X_norm(:,2:end); %sem a colune inicial de '1's
m = size(X_,1);
n = size(X_,2);
X = zeros(m,n);

for i=1:m
    for p=1:n
        X_(i,p) = X_(i,p)*sigma(p) + mu(p);
    end
end

X_norm(:,2:end) = X_;
X = X_norm;
end