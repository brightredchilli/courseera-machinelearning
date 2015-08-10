function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples
n = length(theta); % number of features (includes the intercept column)

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta


hx = X * theta;
gx = sigmoid(hx); % n x 1 vector

lambdas = ones(n, 1) * lambda;
lambdas(1) = 0; % we make the first param 0, so that the theta_0 is not penalized by regularization

reg_cost = (1/(2*m)) * (lambdas' * (theta.^2));
J = -1/m * (y' * log(gx) + (1-y)' * log(1-gx)) + reg_cost;


reg_grad = lambdas/m .* theta;
grad = 1/m * (X' * (gx - y)) + reg_grad;

% =============================================================

end
