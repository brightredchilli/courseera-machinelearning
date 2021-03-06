function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);


% You need to return the following values correctly
J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost function and gradient for collaborative
%               filtering. Concretely, you should first implement the cost
%               function (without regularization) and make sure it is
%               matches our costs. After that, you should implement the
%               gradient and use the checkCostFunction routine to check
%               that the gradient is correct. Finally, you should implement
%               regularization.
%
% Notes: X - num_movies  x num_features matrix of movie features
%        Theta - num_users  x num_features matrix of user features
%        Y - num_movies x num_users matrix of user ratings of movies
%        R - num_movies x num_users matrix, where R(i, j) = 1 if the
%            i-th movie was rated by the j-th user
%
% You should set the following variables correctly:
%
%        X_grad - num_movies x num_features matrix, containing the
%                 partial derivatives w.r.t. to each element of X
%        Theta_grad - num_users x num_features matrix, containing the
%                     partial derivatives w.r.t. to each element of Theta
%

Y_pred = X * Theta';
Y_pred_error2 = (Y_pred - Y).^2;

J = sum(sum(Y_pred_error2 .* R))/2;

J += (lambda/2) * sum(sum(X.^2)) + (lambda/2) * sum(sum(Theta.^2));

% m = number of users
% v = number of movies
% n = number of features

for i = 1:num_movies
	R_i = R(i,:)'; % who has rated movie i, as a column vector. m x 1

	R_idx = find(R_i); % indexes of who has rated movie i as 1, m_i x 1 column vector

	Theta_i = Theta(R_idx, :); % this is an m_i x n matrix.

	Y_i = Y(i, R_idx)'; % ground truths from users who have rated movie i. This should be m_i x 1

	X_i = X(i, :)'; % feature vector for movie i, should be n x 1

	error_m_i = (Theta_i * X_i - Y_i);

	% error_m_i' is 1 x m_i, Theta_i is m_i x n. Multiply to get 1 x n, which is the proper form for X_grad.
	X_grad(i,:) = error_m_i' * Theta_i;

	% error_m_i is m_i x 1, X_i' is 1 x n. Multiply to get m_i x n,
	%which must then be updated in the proper rows using R
	Theta_grad(R_idx, :) += error_m_i * X_i';

	X_grad(i,:) += lambda * X_i';
end

% because of the way we calculate Theta_grad, we need to do this outside of the forloop,
% otherwise we will have 'repeat' summations in Theta_grad
Theta_grad += lambda * Theta;

% =============================================================

grad = [X_grad(:); Theta_grad(:)];

end
