addpath 'cifar-10-batches-mat';
%% Reading data and initialize the parameters of the network
clear all;
[trainX, trainY, trainy] = LoadBatch('data_batch_1');              %Data for training
[valX, valY, valy] = LoadBatch('data_batch_2');                    %Data for validation
[testX, testY, testy] = LoadBatch('test_batch.mat');               %For Testing
fprintf('Batches loaded \n'); 

trainX = reshape(trainX,3072,10000);    %trainX with size dxn  -> 3072x10000
valX = reshape(valX,3072,10000);        %valX with size dxn  -> 3072x10000
testX = reshape(testX,3072,10000);      %testX with size dxn  -> 3072x10000
d=size(trainX,1); 
%More prepocessing
mean_trainX = mean(trainX, 2);
trainX = trainX - repmat(mean_trainX, [1, size(trainX, 2)]);
fprintf('Preprocessing done \n'); 

%Subtraction from validated and tested data
valX = valX - repmat(mean_trainX, [1, size(valX, 2)]);
testX = testX - repmat(mean_trainX, [1, size(testX, 2)]);
fprintf('Substractions done "----->check<-------" \n');
%% Initiating W1, W2, b1 and b2
m=50; K-10 
[W1,W2,b1,b2] = InitParams(d,m);
fprintf('Initialization done \n');

% -------> Termina Ex1

%%    -------> Beg of Ex2
[P,h,s1] = EvaluateClassifier(trainX, cell2mat(W1), cell2mat(W2), cell2mat(b1), cell2mat(b2));
fprintf('Evaluations done P output \n');

%%   This is for the cost function
lambda=0;
J = ComputeCost(P,trainY,cell2mat(W1),cell2mat(W2),lambda);
fprintf(' cost done \n');
%% Accuracy
acc = ComputeAccuracy(trainX,trainY,P);
fprintf(' accuracy done \n');
%% Gradients
[LW1,LW2,Lb1,Lb2,JW1,JW2,Jb1,Jb2] = ComputeGradients(trainX, trainY, P, cell2mat(W1), cell2mat(W2), h, s1, lambda)
fprintf(' gradients done \n');

