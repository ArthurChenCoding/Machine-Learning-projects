function centroids = kMeansInitCentroids(X, K)
%KMEANSINITCENTROIDS This function initializes K centroids that are to be 
%used in K-Means on the dataset X
%   centroids = KMEANSINITCENTROIDS(X, K) returns K initial centroids to be
%   used with the K-Means on the dataset X
%

% You should return this values correctly
centroids = zeros(K, size(X, 2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should set centroids to randomly chosen examples from
%               the dataset X
%

% Number of training examples
m = size(X,1);

% Generate a randomized array ranging from 1 to m
randidx = randperm(m);

% Choosing first K training examples from X into centroids
centroids = X(randidx(1:K), :);

% =============================================================

end

