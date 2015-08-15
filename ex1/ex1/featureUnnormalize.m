function [X] = featureNormalize(X_norm, mu, sigma, ignore_first_row)

	X_norm = X
	if ignore_first_row
		X_norm = X(:, (2:end));
	end

	n = size(X_norm, 2); % number of features
	m = size(X_norm, 1); % number of datapoints
	X = zeros(n, m);

	for i = 1:n
		featureN = X_norm(:, i);
		X(:, i) = (featureN - mu(i))/sigma(i);
	end

	if ignore_first_row
		X = [X_norm]
	end



end