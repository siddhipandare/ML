function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Computes cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

J = 0;
grad = zeros(size(theta));

% ==============================================================
h = sigmoid(X*theta);
J=((-1/m) * (sum(y.*log(h) + (1.-y).*log(1.-h)))) + (lambda/(2*m))* sum(theta.^2) - (lambda/(2*m))* sum(theta(1).^2);
grad=((1/m)*sum((h-y).* X,1))' + (lambda/m).*(theta) ;
grad(1)= grad(1) -((lambda/m)*(theta(1)));

% =============================================================

grad = grad(:);

end
