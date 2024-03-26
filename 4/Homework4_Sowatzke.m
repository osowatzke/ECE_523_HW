% Clear workspace
clear;

% Set random number generator seed for repeatability
rng('default');

% Load IRIS training data
C = readcell('IRIS_data.xlsx');
data = cell2mat(C(:,1:4));
labels = C(:,5);

% Reformat class labels
labels = categorical(labels);

% Determine the number of unique classes
classes = unique(labels);

% Reformat data
data = data(:,3:4).';
data = reshape(data,size(data,1),1,1,[]);

% Divide data into training and test sets
cv = cvpartition(labels,'KFold',5);
isTraining = cv.training(1);
isTest = cv.test(1);

% Get labels and data for training sets
trainingLabels = labels(isTraining);
trainingData = data(:,:,:,isTraining);

% Get labels and data for test sets
testLabels = labels(isTest);
testData = data(:,:,:,isTest);

% Construct neural network
layers = [
    imageInputLayer([size(data,1),1,1]);
    fullyConnectedLayer(5);
    reluLayer;
    fullyConnectedLayer(3);
    softmaxLayer;
    classificationLayer];

% Setup neural network training options
opts = trainingOptions('adam',...
    'Shuffle','every-epoch',...
    'LearnRateSchedule','piecewise',...
    'LearnRateDropPeriod',100,...
    'LearnRateDropFactor',0.5,...
    'L2Regularization',0.001,...
    'MaxEpochs',100,...
    'MiniBatchSize',8,...
    'ExecutionEnvironment','auto',...
    'Plots','training-progress',...
    'Verbose',true);

% train network
net = trainNetwork(trainingData,trainingLabels,layers,opts);

% Determine how well network performs with training data
Z = predict(net,trainingData);
[~,I] = max(Z,[],2);
predictedLabels = classes(I);

% Create confusion matrix for training data set
confusionMatrix = zeros(length(classes));
for i = 1:length(classes)
    sampleSelect = (trainingLabels == classes(i));
    classPredictions = predictedLabels(sampleSelect);
    confusionMatrix(i,:) = sum(classPredictions == classes.');
end
disp('Confusion Matrix for Training Dataset:')
disp(confusionMatrix)

% Compute classification accuracy for training dataset
% Assume equal prior probabilities
pCorrect =  diag(confusionMatrix)./sum(confusionMatrix,2);
pCorrect = mean(pCorrect);
disp('Classification Accuracy for Training Dataset:')
disp(pCorrect)

% Determine how well network performs with test data
Z = predict(net,testData);
[~,I] = max(Z,[],2);
predictedLabels = classes(I);

% Create confusion matrix for test data set
confusionMatrix = zeros(length(classes));
for i = 1:length(classes)
    sampleSelect = (testLabels == classes(i));
    classPredictions = predictedLabels(sampleSelect);
    confusionMatrix(i,:) = sum(classPredictions == classes.');
end
disp('Confusion Matrix for Test Dataset:')
disp(confusionMatrix)

% Compute classification accuracy for test dataset
% Assume equal prior probabilities
pCorrect =  diag(confusionMatrix)./sum(confusionMatrix,2);
pCorrect = mean(pCorrect);
disp('Classification Accuracy for Test Dataset:')
disp(pCorrect)