
%% Initialization
clear ; close all; clc


input_layer_size  = 11520;  
num_labels = 4;          % 4 labels, from 1 to 4
                

%% =========== Part 1: Loading Data =============
%   loading the dataset.
% 
%
X_all=csvread("FertPredictDataset.csv");

X_train =  X_all(1:1280,1:9);
y_train =  X_all(1:1280,10);

X_test =  X_all(1281:end,1:9);
y_test =  X_all(1281:end,10);

%% ============ Part 2b: One-vs-All Training ============
fprintf('\nTraining One-vs-All Logistic Regression...\n')

lambda = 0.1;
[all_theta] = oneVsAll(X_train, y_train, num_labels, lambda);

fprintf('Program paused. Press enter to continue.\n');
pause;


%% ================ Part 3: Predict for One-Vs-All ================

pred = predictOneVsAll(all_theta, X_test);

fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y_test)) * 100);






