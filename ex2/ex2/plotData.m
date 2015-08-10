function plotData(X, y)
%PLOTDATA Plots the data points X and y into a new figure 
%   PLOTDATA(x,y) plots the data points with + for the positive examples
%   and o for the negative examples. X is assumed to be a Mx2 matrix.

% Create New Figure
figure; hold on;

% ====================== YOUR CODE HERE ======================
% Instructions: Plot the positive and negative examples on a
%               2D plot, using the option 'k+' for the positive
%               examples and 'ko' for the negative examples.
%


[m, n] = size(X);

X_0 = X(find(y == 0),:); % this gives us all negative examples
X_1 = X(find(y == 1),:); % this gives us all positive examples

plot(X_1(:, 1), X_1(:, 2), 'r+');
plot(X_0(:, 1), X_0(:, 2), 'ko');


xlabel('x1');
ylabel('x2');

legend('Not admitted', 'Admitted');







% =========================================================================



hold off;

end
