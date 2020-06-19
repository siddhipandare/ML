function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);


p = zeros(size(X, 1), 1);


X=[ones(m,1) X];
z2= Theta1 * X';
a2=sigmoid(z2);
n=size(a2,2);
a2=[ones(1,n);a2];
z3=Theta2 * a2;
a3=sigmoid(z3);

[u,p]=max(a3', [], 2);
p=p;





% =========================================================================


end
