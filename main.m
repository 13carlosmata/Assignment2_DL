addpath 'cifar-10-batches-mat';
clear all;
close all;

%% Reading data and initialize the parameters of the network

%Best fit should be is at eta0.0385 and lambda 0.000001
fprintf('   --> Running Code \n'); 
fprintf('GD parameters '); 
GD=GDparams;
GD.n_batch=100;
GD.n_epochs=30;
GD.eta=0.0385;
lambda=0.000001;            fprintf('- done\n'); 
%%  Loading Batches from CIFAR
fprintf('Loading Batch '); 
%           --Using 1 file--   Start
[trainX, trainY, trainy] = LoadBatch('data_batch_1');              %Data for training
%           --Using 1 file--   End

%           --Using 5 files--   Start
% [trainX1, trainY1, trainy1] = LoadBatch('data_batch_1');              %Data for training
% [trainX2, trainY2, trainy2] = LoadBatch('data_batch_2');              %Data for training
% [trainX3, trainY3, trainy3] = LoadBatch('data_batch_3');              %Data for training
% [trainX4, trainY4, trainy4] = LoadBatch('data_batch_4');              %Data for training
% [trainX5, trainY5, trainy5] = LoadBatch('data_batch_5');              %Data for training
%           --Using 5 files--   End
[valX, valY, valy] = LoadBatch('data_batch_2');                    %Data for validation
[testX, testY, testy] = LoadBatch('test_batch.mat');               %For Testing
fprintf('- done \n');  fprintf('Preprocessing data '); 
%           --Using 1 file--   Start
trainX = reshape(trainX,3072,10000);    %trainX with size dxn  -> 3072x10000
%           --Using 1 file--   End

%           --Using 5 files--   Start
% trainX1 = reshape(trainX1,3072,10000);    %trainX with size dxn  -> 3072x10000
% trainX2 = reshape(trainX2,3072,10000);    %trainX with size dxn  -> 3072x10000
% trainX3 = reshape(trainX3,3072,10000);    %trainX with size dxn  -> 3072x10000
% trainX4 = reshape(trainX4,3072,10000);    %trainX with size dxn  -> 3072x10000
% trainX5 = reshape(trainX5,3072,10000);    %trainX with size dxn  -> 3072x10000
% trainX=[trainX1,trainX3,trainX4,trainX5];
% trainY=[trainY1,trainY3,trainY4,trainY5];
%           --Using 5 files--   End

valX = reshape(valX,3072,10000);        %valX swith size dxn  -> 3072x10000
testX = reshape(testX,3072,10000);      %testX with size dxn  -> 3072x10000
d=size(trainX,1); 


%More prepocessing
mean_trainX = mean(trainX, 2);
trainX = trainX - repmat(mean_trainX, [1, size(trainX, 2)]);
fprintf('- done\n');      fprintf('Running substractions with mean_X '); 
%Subtraction from validated and tested data
valX = valX - repmat(mean_trainX, [1, size(valX, 2)]);
testX = testX - repmat(mean_trainX, [1, size(testX, 2)]);
fprintf('- done\n');
%% Initiating W1, W2, b1 and b2
fprintf('Initialization of W{} and b{} ');
m=50; K=10;
[W,b] = InitParams(d,m,K); fprintf('- done \n');
%%    -------> Beg of Ex2
fprintf('Evaluating Classifier ');
[P,h,s1] = EvaluateClassifier(trainX, W, b);        fprintf('- done \n');
%%   This is for the cost function
fprintf('Computing Cost ');
[J,J1] = ComputeCost(trainX,trainY,W,b,lambda);     fprintf('- done \n');
fprintf('      > Initial Loss = %f\n', J);
%% Accuracy
fprintf('Computing Accuracy ');
acc = ComputeAccuracy(trainX,trainY,P);             fprintf('- done \n');
%% Gradients
fprintf('Computing Gradients ');
[LW,Lb,JW,Jb] = ComputeGradients(trainX, trainY, P, W,b, h, s1, lambda);    fprintf('- done \n');

%% Evaluation of gradients
% 
% [LW,Lb,JW,Jb] = ComputeGradients(trainX(:,1), trainY(:,1), P, W,b, h, s1, lambda);
% fprintf('    *Analytical calculated \n');
% 
% hexr = 1e-5;
% [grad_b, grad_W] = ComputeGradsNum(trainX(:,1), trainY(:,1), W,b, lambda, hexr);
% fprintf('    *Numerical calculated \n');
% fprintf('Numerical-Analitical comparison done\n');
% diff_LW1=abs(max(max(LW{1}-grad_W{1})));
% diff_LW2=abs(max(max(LW{2}-grad_W{2})));
% diff_Lb1=abs(max(Lb{1}-grad_b{1}));
% diff_Lb2=abs(max(Lb{2}-grad_b{2}));
% 
% fprintf('  Worst approximation for W1: %d\n',diff_LW1);
% fprintf('  Worst approximation for W2: %d\n',diff_LW2);
% fprintf('  Worst approximation for b1: %d\n',diff_Lb1);
% fprintf('  Worst approximation for b1: %d\n',diff_Lb1);

%% Working with minibatches   -- This one is using one part of the data

%  [W1star,W2star,b1star,b2star,JK] = MiniBatchGD(trainX(:,1:700), trainY(:,1:700), GD, cell2mat(W1),cell2mat(W2),cell2mat(b1),cell2mat(b2), lambda);
%  fprintf(' minibatch done \n');
%  figure;
%  plot(JK);
% [P,h,s1] = EvaluateClassifier(trainX(:,1:700), W1star, W2star, b1star, b2star);
% accN = ComputeAccuracy(trainX(:,1:700),trainY(:,1:700),P);
% fprintf('Initial Loss = %d\n', J);

%% Working with the full batch of data
fprintf('Using MiniBatch ');
[Wstar,bstar,JK] = MiniBatchGD(trainX, trainY, GD, W, b, lambda);   fprintf('- done \n');
%figure; plot(0:GD.n_epochs,JK);
[Pn,h,s1] = EvaluateClassifier(trainX, Wstar, bstar);
acc_New = ComputeAccuracy(trainX,trainY,Pn);
fprintf(strcat('      > New Accuracy = ', ' ', num2str(acc_New),'%%\n'))
%fprintf('    > New Accuracy = %d\n', acc_New);
%title(strcat('n batch: ',num2str(GD.n_batch),' epochs: ',num2str(GD.n_epochs),' eta: ',num2str(GD.eta),' lambda: ',num2str(lambda)));
%% For Validation Data
fprintf('Computing for Validated Data ');
[Wstar_val,bstar_val,JK_val] = MiniBatchGD(valX, valY, GD, W, b, lambda);
%figure; plot(0:GD.n_epochs,JK_val);
fprintf('- done\n');
figure
[P_val,h_val,s1_val] = EvaluateClassifier(valX, Wstar, bstar);
acc_New_val = ComputeAccuracy(valX,valY,P_val);
fprintf(strcat('      > Accuracy for Validated Data = ', ' ', num2str(acc_New_val),'%%\n'))
%%
plot(0:GD.n_epochs,JK,'r',0:GD.n_epochs,JK_val,'b');
legend(['training loss','     ',num2str(acc_New),'%'],...
    ['validation loss','   ',num2str(acc_New_val),'%']);
title(strcat('Parameters used: ', ' n.batch: ',num2str(GD.n_batch),' epochs: ',num2str(GD.n_epochs),' eta: ',num2str(GD.eta),' lambda: ',num2str(lambda)), 'FontSize',15);
% 
%%
% [P_val,h_val,s1_val] = EvaluateClassifier(valX, Wstar, bstar);
load 'etas';
load 'lambdas';

figure
x = [0:10];
subplot(1,2,1)


plot(x,lambda0_0000001,x,lambda0_000001,x,lambda0_00001,x,lambda0_0001,x,lambda0_001,x,lambda0_01,x,lambda0_1,x,lambda0,x,lambda1,'LineWidth', 1.5)
legend(['lambda = 1e-7'],...
    ['lambda = 1e-6'],...
    ['lambda = 1e-5'],...
    ['lambda = 1e-4'],...
    ['lambda = 1e-3'],...
    ['lambda = 1e-2'],...
    ['lambda = 1e-1'],...
    ['lambda = 0'],...
    ['lambda = 1e+0'])
title('Loss Function at different values of lambda (eta=0.0385)','FontSize',20)
axis([0 10 1 2.5])

subplot(1,2,2)
plot(x,eta0_01,x,eta0_015,x,eta0_03,x,eta0_0385,x,eta0_05,x,eta0_09,x,eta0_1,x,eta0_2,x,eta0,'LineWidth', 1.5)
legend(['eta = 0.01'],...
    ['eta = 0.015'],...
    ['eta = 0.03'],...
    ['eta = 0.0385'],...
    ['eta = 0.05'],...
    ['eta = 0.09'],...
    ['eta = 0.1'],...
    ['eta = 0.2'],...
    ['eta = 0'])
title('Loss Function at different values of eta (lambda=1e-6)','FontSize',20)
axis([0 10 1 2.5])

fprintf('--> Code ran successfully <-- \n');




