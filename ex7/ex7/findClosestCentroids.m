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

distances = [];

for i = 1:K
	c_i = centroids(i, :);
	d = sum(bsxfun(@minus, c_i, X).^2, 2); % m x 1 matrix, distance between c_i and all the sample x
	distances(:, i) = d;
end

% distances is an m x k matrix with all the distances
[minC idx] = min(distances, [], 2); % find the min distance in every row. the column index will be the centroid

% =============================================================

end

