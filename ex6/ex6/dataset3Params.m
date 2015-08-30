function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and
%   sigma. You should complete this function to return the optimal C and
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example,
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using
%        mean(double(predictions ~= yval))
%

CvaluesToTest = [0.01 0.03 0.1 0.3 1 3 10 30];
sigmaValuesToTest = [0.003, 0.01 0.03 0.1 0.3 1 3 10 30];

errorMatrix = zeros(length(CvaluesToTest), length(sigmaValuesToTest));

for i = 1:length(CvaluesToTest)
	for j = 1:length(sigmaValuesToTest)
		curr_C = CvaluesToTest(i);
		curr_sigma = sigmaValuesToTest(j);

		model = svmTrain(X, y, curr_C, @(x1, x2) gaussianKernel(x1, x2, curr_sigma));

		predictions = svmPredict(model, Xval);
		errorMatrix(i,j) = mean(double(predictions ~= yval)); %~=, because == will calculate the accuracy.

	end
end

%find the index of the lowest error in the matrix, that will give us desired C and sigma values

[minError ind] = min(errorMatrix(:)); %convert matrix into vector, find the minimum error and it's index.

[i j] = ind2sub(size(errorMatrix), ind); % find the subscript given a vector and matrix size.


C = CvaluesToTest(i);
sigma = sigmaValuesToTest(j);

fprintf('C = %f and sigma = %f gives us error = %f', C, sigma, minError);


% =========================================================================

end
