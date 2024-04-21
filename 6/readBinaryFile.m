% Function reads data from the binary files which make up
% the MNIST dataset. It takes a file name and the number
% of data dimensions as input arguments
function data = readBinaryFile(fileName, numDims)

    % open file
    fid = fopen(fileName);

    % Read file header
    fread(fid,1,'int32','b');

    % Read data dimensions
    dims = zeros(1,numDims);
    for i = 1:numDims
        dims(i) = fread(fid,1,'int32','b');
    end

    % Ensure dimensions array has at least two elements
    if length(dims) == 1
        dims = [dims, 1];
    end

    % Flip dimensions
    dims = flip(dims);

    % Determine the number of data elements
    numEl = prod(dims);

    % Read data from the file
    data = fread(fid,numEl,'uint8','b');

    % Reshape the data to match the number of dimensions
    data = reshape(data,dims);

    % Close the input file
    fclose(fid);
end