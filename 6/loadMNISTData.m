% Function loads data and labels from the MNIST dataset.
% It takes the dataType ('test' or 'train') as an input
% and returns the corresponding data and training labels
function [data, labels] = loadMNISTData(dataType)

    % Parse the user-provided dataType
    if strcmpi(dataType,'test')
        trainingData = false;
    elseif strcmpi(dataType,'train')
        trainingData = true;
    else
        error(['Unsupported dataType ''%s''. Please select ',...
            'either ''test'' or ''train'''], dataType);
    end

    % Determine the path to the training data set
    pathThisFile = fileparts(mfilename('fullpath'));
    dataSetPath = fullfile(pathThisFile,'MNIST_ORG');

    % Ensure that the training data set exists
    if ~isfolder(dataSetPath)
        error(['Could not find MNIST dataset. Follow the ',...
            'README instructions to download the dataset.']);
    end

    % Determine the names of the input files
    if trainingData
        dataFile = 'train-images.idx3-ubyte';
        labelsFile = 'train-labels.idx1-ubyte';
    else
        dataFile = 't10k-images.idx3-ubyte';
        labelsFile = 't10k-labels.idx1-ubyte';
    end

    % Append paths to the start of the input file names
    dataFile = fullfile(dataSetPath, dataFile);
    labelsFile = fullfile(dataSetPath, labelsFile);

    % Read data from the input files
    data = readBinaryFile(dataFile, 3);
    data = permute(data, [2 1 3]);

    % Read labels from the input files
    labels = readBinaryFile(labelsFile, 1);
end