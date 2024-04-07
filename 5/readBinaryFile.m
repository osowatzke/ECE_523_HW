function data = readBinaryFile(fileName, numDims)
    fid = fopen(fileName);
    fread(fid,1,'int32','b');
    dims = zeros(1,numDims);
    for i = 1:numDims
        dims(i) = fread(fid,1,'int32','b');
    end
    if length(dims) == 1
        dims = [dims, 1];
    end
    dims = flip(dims);
    numEl = prod(dims);
    data = fread(fid,numEl,'uint8','b');
    data = reshape(data,dims);
    fclose(fid);
end