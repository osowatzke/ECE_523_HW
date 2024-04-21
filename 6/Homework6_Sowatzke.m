%% Cleanup workspace
clear;
close all;

%% Load training data
[trainingInput, ~] = loadMNISTData('train');

% permute training data
trainingInput = permute(trainingInput, [1 2 4 3]);

% Get network output
trainingOutput = flip(flip(trainingInput,1),2);

%% Create Network
layers = [
    imageInputLayer([28,28,1]);
    batchNormalizationLayer;
    convolution2dLayer(3,64,'Padding',1);
    reluLayer;
    maxPooling2dLayer(2,'Stride',2,'HasUnpoolingOutputs',true,'Name','mpool1');
    convolution2dLayer(3,128,'Padding',1);
    reluLayer;
    maxPooling2dLayer(3,'Stride',3,'Padding',1,'HasUnpoolingOutputs',true,'Name','mpool2');
    convolution2dLayer(3,16,'Padding',1);
    transposedConv2dLayer(3,128,'Cropping',1);
    reluLayer;
    maxUnpooling2dLayer('Name','unpool2');
    transposedConv2dLayer(3,64,'Cropping',1);
    reluLayer;
    maxUnpooling2dLayer('Name','unpool1');
    transposedConv2dLayer(3,64,'Cropping',1);
    convolution2dLayer(1,1);
    regressionLayer];

lgraph = layerGraph(layers);

% Connect max pooling and unpooling layers
lgraph = connectLayers(lgraph,'mpool2/indices','unpool2/indices');
lgraph = connectLayers(lgraph,'mpool2/size','unpool2/size'); 

lgraph = connectLayers(lgraph,'mpool1/indices','unpool1/indices');
lgraph = connectLayers(lgraph,'mpool1/size','unpool1/size');

%% Train network
% Fix random number generator seed for repeatability
rng('default');

trainingOptions = trainingOptions('adam', ...
    'Shuffle', 'every-epoch',...
    'InitialLearnRate', 1e-3, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropPeriod', 20, ...
    'LearnRateDropFactor', 0.5, ...
    'L2Regularization', 0.001, ...
    'MaxEpochs', 50, ...
    'MiniBatchSize', 128, ...
    'ExecutionEnvironment','auto',...
    'Plots','training-progress',...
    'Verbose', true);

% load('net.mat')
% 
% lgraph = layerGraph(net.Layers);
% 
% % Connect max pooling and unpooling layers
% lgraph = connectLayers(lgraph,'mpool2/indices','unpool2/indices');
% lgraph = connectLayers(lgraph,'mpool2/size','unpool2/size'); 
% 
% lgraph = connectLayers(lgraph,'mpool1/indices','unpool1/indices');
% lgraph = connectLayers(lgraph,'mpool1/size','unpool1/size');

net = trainNetwork(trainingInput,trainingOutput,lgraph,trainingOptions);

%% Load test data
[testInput, ~] = loadMNISTData('test');

% permute training data
testInput = permute(testInput, [1 2 4 3]);

% Get network output
testOutput = flip(flip(testInput,1),2);

measuredOutput = predict(net,testInput);
