%% Setup workspace
clear;
close all;

% Flag specifies whether to do training or not
doTraining = false;

%% Load training data
[trainingInput, trainingLabels] = loadMNISTData('train');

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

% Define network training options
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

if doTraining
    net = trainNetwork(trainingInput,trainingOutput,lgraph,trainingOptions);
else
    load('MNISTNet.mat','net');
end

% Run the training data trough the network
measTrainingOutput = predict(net, trainingInput);

% Get MSE for training data
N = numel(trainingOutput);
err = sum((trainingOutput - measTrainingOutput).^2,'all')/N;

% Display the MSE for the training data set
disp("MSE for Training Dataset:")
disp(err)

%% Load test data
[testInput, testLabels] = loadMNISTData('test');

% permute training data
testInput = permute(testInput, [1 2 4 3]);

% Get reference network output
testOutput = flip(flip(testInput,1),2);

%% Run network on test data
measTestOutput = predict(net,testInput);

%% Display Images from Training and Test Data Set

% Display sample results from training set
f = figure(1);
f.Position = [100, 200, 1050, 400];
clf;
t = tiledlayout(2,5,'TileSpacing','compact');
for i = 1:10
    nexttile;
    idx = find(trainingLabels == (i - 1), 1);
    img = measTrainingOutput(:,:,:,idx);
    imshow(img);
end
title(t, 'Results for Training Dataset')

% Display sample results from test set
f = figure(2);
f.Position = [100, 200, 1050, 400];
clf;
t = tiledlayout(2,5,'TileSpacing','compact');
for i = 1:10
    nexttile;
    idx = find(testLabels == (i - 1), 1);
    img = measTestOutput(:,:,:,idx);
    imshow(img);
end
title(t, 'Results for Test Dataset')

%% Compute MSE for Each Digit

% Compute MSE for training data set
mseTraining = zeros(10,1);
for i = 1:length(mseTraining)
    sampleSelect = (trainingLabels == (i - 1));
    measData = measTrainingOutput(:,:,:,sampleSelect);
    refData = trainingOutput(:,:,:,sampleSelect);
    N = numel(measData);
    mseTraining(i) = sum((measData - refData).^2,'all')/N;
end

% Output MSE for training data set
disp('MSE for Training Dataset:')
fprintf('\n\tDigit%10s\n', 'MSE')
for i = 1:length(mseTraining)
    fprintf('\t%5d%10.2f\n',i-1, mseTraining(i));
end
fprintf('\n');

% Compute MSE for test data set
mseTest = zeros(10,1);
for i = 1:length(mseTest)
    sampleSelect = (testLabels == (i - 1));
    measData = measTestOutput(:,:,:,sampleSelect);
    refData = testOutput(:,:,:,sampleSelect);
    N = numel(measData);
    mseTest(i) = sum((measData - refData).^2,'all')/N;
end

% Output MSE for training data set
disp('MSE for Test Dataset:')
fprintf('\n\tDigit%10s\n', 'MSE')
for i = 1:length(mseTest)
    fprintf('\t%5d%10.2f\n',i-1, mseTest(i));
end
fprintf('\n');

%% Create Histograms of Activation Maps

% Get the activation maps of the bottleneck layer
activationMaps = activations(net,trainingInput,'conv_3');

% Flatten the activation maps
activationMaps = reshape(activationMaps,400,[]);

% Create distribution graphs for the first 5 features
figure(3)
clf;
t = tiledlayout(2,3);
for i = 1:5
    nexttile;
    histogram(activationMaps(i,:));
end
title(t,'Histograms for Activation Map Features')

%% Approximate Distributions of First Five Features

% Compute the mean and variance of the first five features
mu = mean(activationMaps(1:5,:), 2);
sigma = var(activationMaps(1:5,:), 0, 2);

% Display Mean and Variance
fprintf('%10s%10s\n','Mean','Variance')
for i = 1:length(mu)
    fprintf('%10.2f%10.2f\n', mu(i), sigma(i));
end
fprintf('\n');

% Compute the covariance matrix
C = cov(activationMaps(1:5,:).');

% Display the covariance matrix
disp('Covariance Matrix:')
disp(C)
