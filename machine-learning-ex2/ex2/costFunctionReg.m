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

temp = -y.*log(sigmoid(X*theta)) - (1-y).*(log(1- sigmoid(X*theta)));
theta0=theta(1);
temp_1 = sum(theta.^2) - theta0^2;	
J= sum(temp)/m + (lambda/(2*m))*temp_1;

temp2 = (sigmoid(X*theta)- y).*X;
temp_3 = theta;
temp_3(1)=0;
grad= (sum(temp2))'/m + (lambda/m)*temp_3;




% =============================================================

end
