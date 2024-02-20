% Read IRIS data from spreadsheet
IRIS_data = readcell('IRIS_data.xlsx');

% Get the classifier for each row of data
class_truth = double(categorical(IRIS_data(:,end)));

% Create a matrix from the data MxN
IRIS_data = cell2mat(IRIS_data(:,1:(end-1)));

% Find the euclidean distances between each point (Mx1xM matrix)
dist = sum(abs(IRIS_data - permute(IRIS_data, [3 2 1])).^2, 2);

% Remove the singleton dimension to form an MxM matrix
dist = reshape(dist, size(IRIS_data,1), []);

% Entries on the main diagonal are the distance from the point to itself
% Removing these entries is equivalent to training the classifier with all
% samples except the training sample. Since we are looking for minimum
% distances, we can make the main diagonal infinite to avoid selecting
% the entries on the main diagonal
dist(1:(size(IRIS_data,1)+1):end) = Inf;

% Find the nearest neighbor to each sample
[M, I] = min(dist, [], 2);

% Determine the measured class of each sample
class_meas = class_truth(I);

% Create an empty confusion matrix
confusion_matrix = zeros(length(unique(class_truth)));

% Populate each row of the confusion matrix
for i = 1:size(confusion_matrix,1)

    % Select measured class for samples in class i
    meas = class_meas(class_truth == i);

    % determine the number of measurements of each class
    N = sum(meas == (1:size(confusion_matrix,2)));

    % Populate row of confusion matrix
    confusion_matrix(i,:) = N;
end

% Display confusion matrix
disp(confusion_matrix);