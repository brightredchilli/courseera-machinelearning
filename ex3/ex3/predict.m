function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%



a1 = X;
a1_intercept = ones(size(a1, 1), 1); % add bias row
a1 = [a1_intercept a1]; % m x n matrix, where n is the number of nodes in this layer(input layer)

% Theta1 is a s(j) * s(j-1) matrix, where s(j) is the # of nodes in this layer, and s(j-1) is # of nodes from the previous layer
a2 = sigmoid(a1 * Theta1'); 
a2_intercept = ones(size(a2, 1), 1);
a2 = [a2_intercept a2]; % m * n matrix, where n is the numebr of nodes in this layer(hidden layer)

% Theta2 is a s(j) * s(j-1) matrix, where s(j) is the # of nodes in this layer, and s(j-1) is # of nodes from the previous layer
a3 = sigmoid(a2 * Theta2'); % m x k matrix, where k is the number of prediction classes.

% dbstop("predict", 40);

[maxes, p] = max(a3, [], 2);


% =========================================================================


end
