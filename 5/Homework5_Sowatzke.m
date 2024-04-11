% Set random number generator seed for repeatable results
rng('default');

% Load training data
[trainingImages, trainingLabels] = loadMNISTData('train');

% Make categorical array for training labels
trainingLabels = categorical(trainingLabels).';

% Add a single channel dimension to the grayscale images
trainingImages = permute(trainingImages,[1 2 4 3]);

% Get size of training images
[height, width, ~, ~] = size(trainingImages);
imageSize = [height, width, 1];

% Get number of categories in the input image
numImageCategories=length(unique(trainingLabels));

% Create Input Layer
inputLayer = imageInputLayer(imageSize,'Normalization','none');

% Convolutional layer parameters
filterSize = [5 5];
numFilters = 32;

middleLayers = [
    batchNormalizationLayer
    % The first convolutional layer has a bank of 32 5x5 filters. A
    % symmetric padding of 2 pixels is added to ensure that image
    % borders are included in the processing. This is important to
    % avoid information at the borders being washed away too early
    % in the network. The output of this layer is 28x28.
    convolution2dLayer(filterSize, numFilters, 'Padding', 2)

    % Next add the ReLU layer:
    reluLayer()

    % Follow it with a max pooling layer that has a 2x2 spatial
    % pooling area and a stride of 2 pixels. This down-samples the
    % data dimensions from 28x28 to 14x14
    maxPooling2dLayer(2, 'Stride', 2)

    % Add a convolution layer with a symmetric padding of 3 pixels.
    % The output of this layer is 16x16. Adding an additional pixel
    % of symmetric padding to make the layer output a power of 2
    convolution2dLayer(filterSize, numFilters, 'Padding', 3)

    % Add another ReLU layer
    reluLayer()

    % Add a max pooling layer to down-sample the data dimensions
    % from 16x16 to 8x8
    maxPooling2dLayer(2, 'Stride', 2)

    % Repeat the first layers twice to get an 2x2 output layer
    convolution2dLayer(filterSize, 2 * numFilters, 'Padding', 2)
    reluLayer()
    maxPooling2dLayer(2, 'Stride', 2)

    convolution2dLayer(filterSize,  numFilters, 'Padding', 2)
    reluLayer()
    maxPooling2dLayer(2, 'Stride', 2)
];

finalLayers = [

    % Add a fully connected layer with 64 output neurons. The
    % output size of this layer will be an array with a length
    % of 64.
    fullyConnectedLayer(64)
    
    % Add an ReLU non-linearity.
    reluLayer
    
    % Add the last fully connected layer. At this point, the
    % network must produce 10 signals that can be used to measure
    % whether the input image belongs to one category or another.
    % This measurement is made using the subsequent loss layers.
    fullyConnectedLayer(numImageCategories)
    
    % Add the softmax loss layer and classification layer. The
    % final layers use the output of the fully connected layer to
    % compute the categorical probability distribution over the
    % image classes. During the training process, all the network 
    % eights are tuned to minimize the loss over this categorical
    % distribution.
    softmaxLayer
    classificationLayer
];

% Concatenate all the layers to form the full neural network
layers = [
    inputLayer
    middleLayers
    finalLayers
];

% Set the network training options
opts = trainingOptions('adam', ...
    'InitialLearnRate', 1e-3, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.1, ...
    'LearnRateDropPeriod', 20, ...
    'L2Regularization', 0.004, ...
    'MaxEpochs', 5, ...
    'MiniBatchSize', 128, ...
    'ExecutionEnvironment','auto',...
    'Plots','training-progress',...
    'Verbose', true);

% A trained network is loaded from disk to save time when running
% the example. Set this flag to true to train the network.
doTraining = 1;

if doTraining
    % Train a network.
    MNISTNet = trainNetwork(trainingImages, trainingLabels,...
        layers, opts);
    save('MNISTNet.mat', 'MNISTNet');
else
    % Load pre-trained detector for the example.
    load('MNISTNet.mat', 'MNISTNet');
end

% Load test data
[testImages, testLabels] = loadMNISTData('test');

% Make categorical array for training labels
testLabels = categorical(testLabels).';

% Add a single channel dimension to the grayscale images
testImages = permute(testImages,[1 2 4 3]);

% Classify each of the test images
YTest = classify(MNISTNet, testImages);

% Determine and report the accuracy
accuracy = sum(YTest == testLabels)/numel(testLabels);
disp('CNN Accuracy:')
disp(accuracy);

% Determine and report the confusion matrix
confusionMatrix=confusionmat(YTest,testLabels);
disp('CNN Confusion Matrix:')
disp(confusionMatrix);

% Reshape the training images to be a MxN array where M is the
% number of training samples and N is the number of features
X = reshape(trainingImages,[],size(trainingImages,4)).';

% Training labels are used as is
Y = trainingLabels;

% Create a KNN Classifer with K=3
Mdl = fitcknn(X,Y,'NumNeighbors',3);

% Create an MxN array from the test images
X = reshape(testImages,[],size(testImages,4)).';

% Predict the class of the test images using the KNN Classifier
YTest = predict(Mdl, X);

% Determine and report the accuracy
accuracy = sum(YTest == testLabels)/numel(testLabels);
disp('KNN Accuracy:')
disp(accuracy);

% Determine and report the confusion matrix
confusionMatrix=confusionmat(YTest,testLabels);
disp('KNN Confusion Matrix:')
disp(confusionMatrix);