function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

##C_row=[0.01,0.03,0.1,0.3,1,3,10,30];
##sigma_row=[0.01,0.03,0.1,0.3,1,3,10,30];
##
##prederr=7855;
##
##for i=1:length(C_row)
##  for j=1:length(sigma_row)
##    C_check=C_row(i);
##    
##    sigma_check=sigma_row(j);
##    
##    model= svmTrain(X, y, C_check, @(x1, x2) gaussianKernel(x1, x2, sigma_check));
##    
##    predictions = svmPredict(model, Xval);
##    
##    prederr_temp=mean(double(predictions ~= yval));
##    
##    if (prederr_temp < prederr)
##      C=C_check;
##      
##      sigma=sigma_check;
##      
##      prederr=prederr_temp;
##      
##    endif
##    
##    
##    
##  endfor
##endfor
##
##disp(C);
##disp(sigma);
##ctrl+shift+R to uncomment
##ctrl+R to comment

C=0.3 ;
sigma=0.1;







% =========================================================================

end
