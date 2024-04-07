function [data, labels] = loadMNISTData(dataType)

    if strcmpi(dataType,'test')
        trainingData = false;
    elseif strcmpi(dataType,'train')
        trainingData = true;
    else
        error("Unsupported dataType '%s'. Please select either 'test' or 'train'");
    end

    pathThisFile = fileparts(mfilename('fullpath'));
    dataSetPath = fullfile(pathThisFile,'MNIST_ORG');

    if ~isfolder(dataSetPath)
        error('Could not find MNIST dataset. Follow the instructions in the README to download the dataset.');
    end

    if trainingData
        dataFile = fullfile(dataSetPath,'train-images.idx3-ubyte');
        labelsFile = fullfile(dataSetPath,'train-labels.idx1-ubyte');
    else
        dataFile = fullfile(dataSetPath,'t10k-images.idx3-ubyte');
        labelsFile = fullfile(dataSetPath,'t10k-labels.idx1-ubyte');
    end

    data = readBinaryFile(dataFile, 3);
    data = permute(data, [2 1 3]);

    labels = readBinaryFile(labelsFile, 1);
end