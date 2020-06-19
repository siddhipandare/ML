function [error_train, error_test] = ...
    learningCurve(X, y, X_test, y_test, lambda)
%LEARNINGCURVE Generates the train and cross validation set errors needed 
%to plot a learning curve
%   [error_train, error_test] = ...
%       LEARNINGCURVE(X, y, X_test, y_test, lambda) returns the train and
%       cross validation set errors for a learning curve. In particular, 
%       it returns two vectors of the same length - error_train and 
%       error_test. Then, error_train(i) contains the training error for
%       i examples (and similarly for error_test(i)).
%
%   In this function, you will compute the train and test errors for
%   dataset sizes from 1 up to m. In practice, when working with larger
%   datasets, you might want to do this in larger intervals.
%

% Number of training examples
m = size(X, 1);
% You need to return these values correctly
error_train = zeros(m, 1);
error_test   = zeros(m, 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return training errors in 
%               error_train and the cross validation errors in error_test. 
%               i.e., error_train(i) and 
%               error_test(i) should give you the errors
%               obtained after training on i examples.
%
% Note: You should evaluate the training error on the first i training
%       examples (i.e., X(1:i, :) and y(1:i)).
%
%       For the cross-validation error, you should instead evaluate on
%       the _entire_ cross validation set (X_test and y_test).
%
% Note: If you are using your cost function (linearRegCostFunction)
%       to compute the training and cross validation error, you should 
%       call the function with the lambda argument set to 0. 
%       Do note that you will still need to use lambda when running
%       the training to obtain the theta parameters.
%
% Hint: You can loop over the examples with the following:
%
%       for i = 1:m
%           % Compute train/cross validation errors using training examples 
%           % X(1:i, :) and y(1:i), storing the result in 
%           % error_train(i) and error_test(i)
%           ....
%           
%       end
%

% ---------------------- Sample Solution ----------------------

for i =1:m
  X_train= X(1:i,:);
  y_train=y(1:i);
  
  theta= trainLogReg(X_train, y_train, lambda);
  disp(theta)
  disp("^^^")
  
  error_train(i)=lrCostFunction(X_train, y_train, theta,0);
  
  error_test(i)=lrCostFunction(X_test, y_test, theta,0);
  
  
endfor







% -------------------------------------------------------------

% =========================================================================

end
