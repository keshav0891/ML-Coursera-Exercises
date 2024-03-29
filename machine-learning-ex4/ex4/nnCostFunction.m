function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));		% of size 25 * 401
Theta2_grad = zeros(size(Theta2));		% of size 10 * 26

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
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
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

[k,useless]=size(Theta2);
y_new= zeros(m,k);
map= eye(k);
ind=0;

for i=1:m
   ind=y(i);
   if(ind)
	y_new(i,:)=map(ind,:);
   else
	y_new(i,:)=map(k,:);
   endif
endfor 

Ori_theta1=Theta1;
Ori_theta2=Theta2;

X=[ones(m,1) X];		% of dimesnsion m * n+1
z2= X*(Theta1)';		% of dimension m * 25
a2= sigmoid(z2);		% of dimension m * 25
a2=[ones(size(z2,1), 1) a2];	% of dimension m * 26
z3= a2*(Theta2)';		% of dimension m * 10
a3=sigmoid(z3);			% of dimension m * 10


temp= -y_new.*log(a3) - (1-y_new).*log(1-a3);
J= sum(sum(temp,2),1)/m;

%[theta1_m, theta1_n]=size(Theta1)
Theta1(:,1)=[];		% of dimension theta1_m * m
Theta2(:,1)=[];		% of dimension k* theta1_m i.e 10 * 25

sumTheta1= Theta1.*Theta1;
tempTheta1= (lambda*sum(sum(sumTheta1,2),1))/(2*m);

sumTheta2= Theta2.*Theta2;
tempTheta2= (lambda* sum(sum(sumTheta2,2),1))/(2*m);

J= J + tempTheta1 + tempTheta2;

% -------------------------------------------------------------

delta3 = a3 - y_new;		% of dimension m * 10(i.e K)
delta2 = (delta3*Theta2).*sigmoidGradient(z2);		% of dimension m * 25

Theta2_grad = (Theta2_grad + (delta3')*a2)/m;		% 10*26
for i= 2:useless
     Theta2_grad(:,i) = Theta2_grad(:,i) .+ (lambda/m)*Ori_theta2(:,i);
endfor
Theta1_grad = (Theta1_grad + (delta2')*X)/m;		% 25*401
[a,b]=size(Theta1_grad);
for j= 2:b
     Theta1_grad(:,j) = Theta1_grad(:,j) .+ (lambda/m)*Ori_theta1(:,j);
endfor
% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
