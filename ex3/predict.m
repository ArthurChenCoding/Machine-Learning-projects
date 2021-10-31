function p = predict(Theta1, Theta2, X)
%25 * 401
%10 * 26
%5000 * 400

%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1); % 5000
num_labels = size(Theta2, 1); % 10

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1); % 5000 * 1

% Add ones to the X data matrix
X = [ones(m, 1) X]; %5000*401

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

a2 = sigmoid(X * Theta1'); % 5000 * 25 = 5000 * 401 by 401*25

% Add ones to the a2 data matrix
a2 = [ones(m, 1) a2]; % 5000 * 26

a3 = sigmoid(a2 * Theta2'); % 5000 * 10 = 5000 * 26 by 26 * 10

[~, p] = max(a3, [], 2); % obtain Max over the 2nd dimension = 5000 * 1
% return only the 2nd output

% =========================================================================


end
