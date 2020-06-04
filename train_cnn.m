% Framework to train CNN implemented in Matlab by Lilong Shi
% reference: https://houxianxu.github.io/2015/04/25/support-vector-machine/
%            https://ljvmiranda921.github.io/notebook/2017/02/11/multiclass-svm/
clear all;
close all;

use_vectorize = true;

reg = 1e0;
lr = 1e-3;
load TRAIN_X.mat
load TRAIN_Y.mat
method = 'svd';

X = TRAIN_X; % add biases dimension %rand(2500,36);
y = TRAIN_Y+1; %rand(2500,1);

n_filter = 10; d_filter = 1; h_filter = 3; w_filter = 3; stride = 1; padding = 1;
num_classes = 10; % C
num_F1 = 100; % size of hidden layers
num_train = size(X,1); % N
num_dim = size(X,2);
num_epoch = 500;

%% define network
NN = {'IN',num_train,1,6,6 }; % # batch size, channel, height, width, 
NN{end+1} = {'CONV',n_filter,h_filter,w_filter,padding,stride}; % # of filter, height, width, padding, stride
NN{end+1} = {'POOL', 2,padding,2}; % size , padding, stride
NN{end+1} = {'FC', 100}; % hidden layer size
NN{end+1} = {'OUT', num_classes}; % # of classes 

idx = sub2ind([num_train num_classes], (1:length(y)).', y);

%% initialize weights
b0 = zeros(n_filter,1);
W0 = randn(n_filter,d_filter,h_filter,w_filter)*.01;% *sqrt(2/(d_filter*h_filter*w_filter)) ;
W1 = randn(num_dim*n_filter,num_F1)*.01;% *sqrt(2/(num_dim*n_filter)) ;
b1 = zeros(1,num_F1);
W2 = randn(num_F1,num_classes)*.01;%*sqrt(2/num_F1) ;
b2 = zeros(1,num_classes);

% vectorized version
for n = 1:num_epoch

    % evaluate class score 
    [F0,X_col] = conv_forward(reshape(X,[num_train,1,6,6]),W0,b0,stride, padding);
    F0 = max(0,reshape(permute(F0,[1 3 4 2]),num_train,[]));
    F1 = max(0,F0*W1+b1); % activation after 1st layer
    F2 = F1*W2+b2; % result after classification

    if strcmp(method,'svm')
        correct_class_score = F2(idx);
        margin = F2 - repmat(correct_class_score,[1 num_classes]) + 1;
        scores = max(0,margin);
        scores(idx) = 0;
        % compute loss
        loss = sum(scores(:))/num_train;
        loss = loss ;%+ 0.5*reg*sum(sum(W1.*W1)) + 0.5*reg*sum(sum(W2.*W2));
        % compute gradient
        dscores = zeros(size(F2));
        dscores(scores>0) = 1;
        dscores(idx) = -1 * sum(scores>0,2); % compute the number of margin > 0
    else
        F2 = exp(F2);
        scores = F2./repmat(sum(F2,2),[1 num_classes]);
        correct_class_score = scores(idx) ;
        % compute loss
        loss = sum(-log(correct_class_score))/num_train;
        loss = loss ;%+ 0.5*reg*sum(sum(W1.*W1)) + 0.5*reg*sum(sum(W2.*W2));
        % compute gradient
        dscores = scores;
        dscores(idx) = dscores(idx)-1;
    end
    dscores = dscores/num_train;

    % backpropgate the gradient to the parameters (W,b)
    dW2 = F1'*dscores;
    db2 = sum(dscores,1);
    dF1 = dscores*W2';
    dF1(F1<=0) = 0; % ReLU

    % backpro to the 1st layer
    dW1 = F0'*dF1;
    db1 = sum(dF1,1);
    dF0 = dF1*W1';
    dF0(F0<=0) = 0; % ReLU

    dF0 = permute(reshape(dF0,num_train,6,6,n_filter),[1 4 2 3]);
    X_col = im2col_indices(reshape(X,[num_train,1,6,6]), h_filter, w_filter, padding, stride);
    [~,dW0,db0] = conv_backward(dF0, X_col, W0, size(reshape(X,[num_train,1,6,6])), stride, padding);

    % add regularization gradient contribution
    dW2 = dW2 + reg*W2;
    dW1 = dW1 + reg*W1;
    dW0 = dW0 + reg*W0;

    % perform a parameter update
    W0 = W0 - lr *dW0;
    b0 = b0 - lr*db0;
    W1 = W1 - lr *dW1;
    b1 = b1 - lr*db1;
    W2 = W2 - lr *dW2;
    b2 = b2 - lr*db2;

    losses_history(n) = loss;
    fprintf('Train#%d: Loss = %.2f\n',n,loss);
end

figure; plot(losses_history); grid on; xlabel('# of epoch'); ylabel('Loss');

load TEST_X.mat
load TEST_Y.mat

%TEST_X = TRAIN_X;
%TEST_Y = TRAIN_Y;

X = TEST_X ;
Y = TEST_Y+1;

confMat = confusionmat(Y, pred_ys);
sum(diag(confMat))/2500

[F0,~] = conv_forward(reshape(X,[num_train,1,6,6]),W0,b0,stride, padding);
F0 = max(0,reshape(permute(F0,[1 3 4 2]),num_train,[]));
scores = max(0,(F0*W1+b1))*W2+b2 ;
[~,pred_ys] = max(scores,[],2);

confMat = confusionmat(Y, pred_ys);
sum(diag(confMat))/2500
