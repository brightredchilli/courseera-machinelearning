function [a b c nn sizes] = makeTestNN()

% layers
sizes = [3 4 5 3];

a = ones(4, 3) * 9;
a = [ones(4, 1) a];

b = ones(5, 4) * 2;
b = [ones(5, 1) b];

c = ones(3, 5) * 3;
c = [ones(3, 1) c];

nn = [a(:); b(:); c(:)];

end
