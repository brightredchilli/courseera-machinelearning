function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %

    % theta is a n+1 column vector
    % X is a m x n+1 matrix
    % alpha is the learning parameter

    % we update theta with the algorithm theta(i) := -alpha * 1/m * sum_1_m((h(x_i) - y) * x_i)

    theta = theta - alpha/m * (X' * (X * theta - y));

    

    % ============================================================

    % Save thes cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
