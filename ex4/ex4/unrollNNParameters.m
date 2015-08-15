function Theta = unrollNNParameters(nn_params, size_of_layers, index)

% Theta = unrollNNParameters(nn_params, size_of_layers, index) returns a Theta matrix of a neural network.
% nn_params The neural network parameters, all rolled up.
% size_of_layers A vector of the sizes of all of the layers, including the input and output layers.
% index The 1-indexed based index of the Theta matrix we would like to retrieve.

% Get offsets

[start_offset end_offset Theta_rows Theta_cols] =  offsetInNNParameters(nn_params, size_of_layers, index);

% Reshape the matrix

Theta = reshape(nn_params(start_offset:end_offset), Theta_rows, Theta_cols);

end
