function [start_offset end_offset Theta_rows Theta_cols] = offsetInNNParameters(nn_params, size_of_layers, index)

L = length(size_of_layers);

if index > L - 1
	error("Requested index(%d) is out-of bounds of the neural-network parameter matrix with %d layers", index, L);
end

if L < 2
	error("Must have at least two layer to unroll");
end

% Find the start offset

start_offset = 1;
for i = 1:index-1
	rows = size_of_layers(i + 1);
	cols = size_of_layers(i) + 1 ; % add bias unit for current layer

	sizeToAdd = rows * cols;
	% fprintf("i = %d rows = %d, cols = %d, total = %d\n",i , rows, cols, sizeToAdd);
	start_offset += sizeToAdd;
end

% Find the end offset

Theta_rows = size_of_layers(index + 1);
Theta_cols = size_of_layers(index) + 1; % always add bias unit for the current layer
Theta_size = Theta_rows * Theta_cols;

end_offset = start_offset + Theta_size - 1; % - 1 to account for 1-indexing

end
