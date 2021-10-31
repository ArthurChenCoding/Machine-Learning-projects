function [J, grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)

% ====================== Introduction ======================

%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.

% ====================== Set up ======================

% dimensions of the input parameters
% input_layer_size  = 400;  
% hidden_layer_size = 25;  
% num_labels = 10;          
                           
% X: 5000 * 400
% y: 5000 * 1
% lambda: 0

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));
% Theta1: 25 * 401
% Theta2: 10 * 26

% Setup some useful variables
m = size(X, 1); %5000
         
% You need to return the following variables correctly 
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== Part 1 ======================
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m

% tranfering y into new form, which only contains 1 and 0 to indicate label.
% y: 5000 * 1 => y_new: 5000 * 10
y_new = zeros(m, num_labels); % 5000 * 10
for i=1:m
  y_new(i, y(i))=1;
end

X = [ones(m, 1), X]; % 5000 * 401
a2 = sigmoid(X * Theta1'); % 5000 * 401 by 401 * 25 = 5000 * 25
a2 = [ones(m, 1), a2]; % 5000 * 26
a3 = sigmoid(a2 * Theta2'); % 5000 * 26 by 26 * 10 = 5000 * 10

%If A is a matrix, then sum(A) returns a row vector containing the sum of each column.
% -y_new'.*(log(a3))' => 10 * 5000 .* 10 * 5000
J = (1/m)*  sum(sum( ((-y_new'.*(log(a3))') - (1-y_new)'.*(log(1-a3))') )); % 1 * 1
Reg = (lambda/(2*m))*( sum(sum(Theta1(:,2:end).^2)) + sum(sum(Theta2(:,2:end).^2)) );

%outputing
J = J + Reg;

% ====================== Part 2 ======================

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

% Back propagation
y_new = y_new';
for t=1:m
    % Step 1
	a1 = X(t,:); % 1 * 401
    a1 = a1';
    z2 = Theta1 * a1;
	a2 = sigmoid(z2); % (1 * 25 = 1 * 401 by 401 * 25)'
    
    a2 = [1; a2]; % 26*1
    z3 = Theta2 * a2;
	a3 = sigmoid(z3); % 10*1 = 10*26 by 26*1
    
    % Step 2
	delta_3 = a3 - y_new(:,t); % (1 * 10 = 1 * 10 - 1 * 10)'
	
    z2 = [1; z2]; % 26*1
    
    % Step 3
    delta_2 = (Theta2' * delta_3) .* sigmoidGradient(z2); % (1*26 = 1*10 by 10*26)'

    % Step 4
	delta_2 = delta_2(2:end); % skipping sigma2(0) (25*1)

	Theta2_grad = Theta2_grad + delta_3 * a2'; % 10*26 = 10*26 + 10*1 by 1*26
	Theta1_grad = Theta1_grad + delta_2 * a1'; % 25*401 = 25*1 by 1*401
end

% Step 5
% remember to match the dimensions with the Theta
Theta1_grad = (1/m) * Theta1_grad; % 25*401
Theta2_grad = (1/m) * Theta2_grad; % 10*26

% ====================== Part 3 ======================

% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.

Theta1_grad(:, 2:end) = Theta1_grad(:, 2:end) + ((lambda/m) * Theta1(:, 2:end));
Theta2_grad(:, 2:end) = Theta2_grad(:, 2:end) + ((lambda/m) * Theta2(:, 2:end));
% ====================== Closing ======================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
