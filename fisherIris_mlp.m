%% MLP - dataset: Fisheriris
%author = @leilamr

close all; 
clear all;
clc

%load dataset
load iris_dataset irisInputs irisTargets

%inputs data
inputs = irisInputs;
targets = irisTargets;

%% create neural network

hiddenLayerSize = 4;
trainFcn = 'trainlm';

net = patternnet(hiddenLayerSize, trainFcn);

% set neural network 
net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 15/100;

net.trainParam.epochs = 1000; 
net.trainParam.max_fail = 500;

net.trainParam.min_grad = 0.000000000000001;

net.trainParam.lr = 0.1;

net.layers{1}.transferFcn='logsig';
net.layers{2}.transferFcn='purelin';

%% train network
[net, tr] = train(net,inputs,targets);

%% test the Network

outputs = net(inputs);
e = gsubtract(targets,outputs);

performance = perform(net,targets,outputs);

tind = vec2ind(targets);
yind = vec2ind(outputs);
percentErrors = sum(tind ~= yind)/numel(tind);

acc = 100 * (1 - percentErrors);


fprintf('Accuracy = %.3f%%', acc);
% View the Network
%view(net)
