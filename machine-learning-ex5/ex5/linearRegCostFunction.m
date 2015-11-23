function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

prediction= X*theta;
theta_reg=theta;
theta_reg([1],:)=[];		%removing first row from the matrix
J=(1/(2*m))*sum((prediction-y).^2) + (lambda/(2*m))*sum(theta_reg.^2);

%[a b]=size(grad);
%[x y]=size(X);
%X(:,1).*(prediction-y)
%X(:,2:y).*(prediction-y)
%(lambda/m)*theta_reg
%x_reg=X;
%x_reg(:,[1])=[];
%x_first=X;
%x_first(:,[2:y])=[];
%(x_first')*(prediction-y)
%(x_reg')*(prediction-y)
%grad(1) = (1/m) * sum((x_first).*(prediction-y));
%grad(2:a) = ((1/m) * sum((x_reg).*(prediction-y))') + (lambda/m)*theta_reg;

theta1=theta;
theta1(1)=0;
grad= (1/m)*(X'*(prediction-y) + lambda*theta1);

% =========================================================================

%grad = grad(:);

end
