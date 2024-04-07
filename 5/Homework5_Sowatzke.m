% Set random number generator seed for repeatable results
rng('default');

% Load training data
[trainingImages, trainingLabels] = loadMNISTData('train');
trainingLabels = categorical(trainingLabels).';
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
    % The first convolutional layer has a bank of 32 5x5x3 filters. A
    % symmetric padding of 2 pixels is added to ensure that image borders
    % are included in the processing. This is important to avoid
    % information at the borders being washed away too early in the
    % network.
    convolution2dLayer(filterSize, numFilters, 'Padding', 2)

    % Note that the third dimension of the filter can be omitted because it
    % is automatically deduced based on the connectivity of the network. In
    % this case because this layer follows the image layer, the third
    % dimension must be 3 to match the number of channels in the input
    % image.

    % Next add the ReLU layer:
    reluLayer()

    % Follow it with a max pooling layer that has a 3x3 spatial pooling area
    % and a stride of 2 pixels. This down-samples the data dimensions from
    % 32x32 to 15x15.
    maxPooling2dLayer(2, 'Stride', 2)

    % Repeat the 3 core layers to complete the middle of the network.
    convolution2dLayer(filterSize, numFilters, 'Padding', 3)
    reluLayer()
    maxPooling2dLayer(2, 'Stride', 2)

    convolution2dLayer(filterSize, 2 * numFilters, 'Padding', 2)
    reluLayer()
    maxPooling2dLayer(2, 'Stride', 2)

    convolution2dLayer(filterSize,  numFilters, 'Padding', 2)
    reluLayer()
    maxPooling2dLayer(2, 'Stride', 2)
];

finalLayers = [

    % Add a fully connected layer with 64 output neurons. The output size of
    % this layer will be an array with a length of 64.
    fullyConnectedLayer(64)
    
    % Add an ReLU non-linearity.
    reluLayer
    
    % Add the last fully connected layer. At this point, the network must
    % produce 10 signals that can be used to measure whether the input image
    % belongs to one category or another. This measurement is made using the
    % subsequent loss layers.
    fullyConnectedLayer(numImageCategories)
    
    % Add the softmax loss layer and classification layer. The final layers use
    % the output of the fully connected layer to compute the categorical
    % probability distribution over the image classes. During the training
    % process, all the network weights are tuned to minimize the loss over this
    % categorical distribution.
    softmaxLayer
    classificationLayer
];

layers = [
    inputLayer
    middleLayers
    finalLayers
    ];

layers(3).Weights = 0.0001 * randn([filterSize 1 numFilters]);

% Set the network training options
opts = trainingOptions('adam', ...
    'InitialLearnRate', 1e-3, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.1, ...
    'LearnRateDropPeriod', 20, ...
    'L2Regularization', 0.004, ...
    'MaxEpochs', 50, ...
    'MiniBatchSize', 128, ...
    'ExecutionEnvironment','auto',...
    'Plots','training-progress',...
    'Verbose', true);

% A trained network is loaded from disk to save time when running the
% example. Set this flag to true to train the network.
doTraining = 0;

if doTraining
    % Train a network.
    MNISTNet = trainNetwork(trainingImages, trainingLabels, layers, opts);
    save('MNISTNet.mat', 'MNISTNet');
else
    % Load pre-trained detector for the example.
    %load('rcnnStopSigns.mat','cifar10Net')
    load('MNISTNet.mat', 'MNISTNet');
end

% Load test data
[testImages, testLabels] = loadMNISTData('test');
testLabels = categorical(testLabels).';
testImages = permute(testImages,[1 2 4 3]);
YTest = classify(MNISTNet, testImages);
accuracy = sum(YTest == testLabels)/numel(testLabels)
CM=confusionmat(YTest,testLabels)