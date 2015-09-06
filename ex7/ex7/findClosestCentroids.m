function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% number of samples
m = size(X,1)

% Set K
K = size(centroids, 1);

% You need to return the following variables correctly.
idx = zeros(m, 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%

X_1 = X(:, 1); % array of all 'x' in the X samples
X_2 = X(:, 2); % array of all 'y' in the X samples

centroids_1 = centroids(:, 1); % array of all 'x' in the centroids
centroids_2 = centroids(:, 2); % array of all 'y' in the centroids

Dx = bsxfun(@minus, X_1, centroids_1'); % m x k matrix of all x differences
Dy = bsxfun(@minus, X_2, centroids_2'); % m x k matrix of all y differences

% first column is distance between c1 and all xs,
% last column is distance between ck and all xs
D =  sqrt(Dx.^2 + Dy.^2);

% we are finding the min of the columns,
% this returns a column vector of the shortest distance between x and all c1-ck
[maxD idx] = min(D, [], 2);

% =============================================================

end

