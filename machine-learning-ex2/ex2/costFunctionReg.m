function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

% cost function
for i = 1:m
    hypot = theta' * X(i, :)';
    J += -y(i) .* log(sigmoid(hypot)) - (1 - y(i)) .* log(1 - sigmoid(hypot));
endfor
J = J ./ m;
regularizationTerm = sum(theta(2:end) .^ 2) .* lambda ./ (2 .* m);
J += regularizationTerm;

% gradient for j >= 1
for j = 2:length(grad)
    sum = 0;
    for i = 1:m
        hypot = theta' * X(i, :)';
        sum += (sigmoid(hypot) - y(i)) .* X(i, j); 
    endfor
    grad(j) = sum ./ m + lambda ./ m .* theta(j);
endfor

% gradient for j == 0
sum = 0;
for i = 1:m
    hypot = theta' * X(i, :)';
    sum += (sigmoid(hypot) - y(i)) .* X(i, 1); 
endfor
grad(1) = sum ./ m; 

% =============================================================

end
