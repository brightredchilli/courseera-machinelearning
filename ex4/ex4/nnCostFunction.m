function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices.
%
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network



# Come up with our own Theta1 and Theta2 using unrollNNParameters
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));


layer_sizes = [input_layer_size hidden_layer_size num_labels];
L = length(layer_sizes);

Theta1 = unrollNNParameters(nn_params, layer_sizes, 1);
Theta2 = unrollNNParameters(nn_params, layer_sizes, 2);

% Setup some useful variables
m = size(X, 1);

% You need to return the following variables correctly
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m

% construct an matrix of answers based on y, which returns us the results as a 0-9 scalar.
% assumes that y is a column vector of the labels
answer = eye(num_labels)(y, :);


X1 = X;
for i = 1:L - 1
	X1 = [ones(size(X1,1), 1) X1]; % add intercept column
	curr_theta = unrollNNParameters(nn_params, layer_sizes, i);
	X1 = sigmoid(X1 * curr_theta'); % get the activation for this layer.'
end

% at this point, X should be a m x 10 matrix

gx = X1; % no need to run sigmoid again as it is alredy done as part of feedforward
J = -1/m * (answer .* log(gx) + (1-answer) .* log(1-gx));
J = sum(sum(J)); % sum up all the errors from all the K labels

%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the
%               first time.


a1 = X;
a1 = [ones(m, 1) a1]; % add intercept

z2 = a1 * Theta1'; %'
a2 = sigmoid(z2);

a2 = [ones(m, 1) a2]; % add intercept
z3 = a2 * Theta2'; %'
a3 = sigmoid(z3);

d3 = a3 .- answer;
d2 = d3 * Theta2(:, 2:end) .* sigmoidGradient(z2);

Theta1_grad = (d2' * a1) / m;
Theta2_grad = (d3' * a2) / m;


% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

Theta1_grad(:, 2:end) += lambda/m * Theta1(:, 2:end);
Theta2_grad(:, 2:end) += lambda/m * Theta2(:, 2:end);

reg_cost = 0;
reg_term = lambda/(2*m);
for i = 1:L - 1
	curr_theta = unrollNNParameters(nn_params, layer_sizes, i);
	curr_theta = curr_theta(:, 2:end); % dont regularize the bias nodes
	curr_theta .^= 2;
	reg_cost += sum(sum(curr_theta)) * reg_term;
end

J += reg_cost;


















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
