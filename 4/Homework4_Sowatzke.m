clear;
close all;
rng('default');
C = readcell('IRIS_data.xlsx');
data = cell2mat(C(:,1:4));
labels = C(:,5);

labels = categorical(labels);
classes = unique(labels);

data = data(:,3:4).';
data = reshape(data,size(data,1),1,1,[]);

cv = cvpartition(labels,'KFold',5);
isTraining = cv.training(1);
isTest = cv.test(1);

% numTrainingSamplesPerClass = 40;
% isTrainingSample = false(size(data,4),1);
% for i = 1:length(classes)
%     classIdx = (1:length(labels)).'.*(labels == classes(i));
%     classIdx = classIdx(classIdx ~= 0);
%     classIdx = classIdx(randperm(length(classIdx)));
%     isTrainingSample(classIdx(1:numTrainingSamplesPerClass)) = true;
% end
% isTestSample = ~isTrainingSample;

trainingLabels = labels(isTraining);
trainingData = data(:,:,:,isTraining);

testLabels = labels(isTest);
testData = data(:,:,:,isTest);

layers = [
    imageInputLayer([size(data,1),1,1]);
    fullyConnectedLayer(5);
    reluLayer;
    fullyConnectedLayer(3);
    softmaxLayer;
    classificationLayer];

opts = trainingOptions('adam',...
    'Shuffle','every-epoch',...
    'LearnRateSchedule','piecewise',...
    "LearnRateDropPeriod",100,...
    'LearnRateDropFactor',0.5,...
    'L2Regularization',0.001,...
    'MaxEpochs',100,...
    'MiniBatchSize',4,...
    'ExecutionEnvironment','auto',...
    'Plots','training-progress',...
    'Verbose',true);

net = trainNetwork(trainingData,trainingLabels,layers,opts);
Z = predict(net,testData);

[~,I] = max(Z,[],2);
predictedLabels = classes(I);

confusionMatrix = zeros(length(classes));
for i = 1:length(classes)
    classPredictions = predictedLabels(testLabels == classes(i));
    confusionMatrix(i,:) = sum(classPredictions == classes.');
end
disp(confusionMatrix)