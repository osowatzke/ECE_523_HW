% Read IRIS data from spreadsheet
IRIS_data = readcell('IRIS_data.xlsx');

% Get the classifier for each row of data
class_truth = double(categorical(IRIS_data(:,end)));

% Create a matrix from the data MxN
IRIS_data = cell2mat(IRIS_data(:,1:(end-1)));

% Combine samples from the second and third classes
class_truth = min(class_truth,2*ones(size(class_truth)));

% Create cell array with data from each class
temp = cell(1,2);
for i = 1:length(temp)
    temp{i} = IRIS_data(class_truth == i,:);
end
IRIS_data = temp;

% Compute the mean and covariance of the IRIS data
m = cellfun(@(x) mean(x).', IRIS_data,'UniformOutput',false);
S = cellfun(@(x) cov(x), IRIS_data,'UniformOutput',false);
S = reshape(S,1,1,length(S));

% Compute Fisher discriminant
Sw = sum(cell2mat(S),3);
w = Sw^(-1)*(m{1} - m{2});

% Project the data onto the discriminant
Y = cellfun(@(X)X*w, IRIS_data, 'UniformOutput', false);

% Project the means onto the discriminant
m = cellfun(@(m) w'*m, m);

% Compute the standard deviation of the projected data
s = cellfun(@(Y) std(Y, 1), Y);

% Assume equal prior probabilities
P = [0.5; 0.5];

% Assume the data is Guassian and pick the threshold which produces
% minimum error. Decision boundary will be quadratic. Store
% resulting polynomial coefficients in an array.
p = conv([1, -m(2)],[1, -m(2)])/(2*s(2)^2) ...   % (x-m2^2)/(2*s2^2)
    - conv([1, -m(1)],[1, -m(1)])/(2*s(1)^2) ... % (x-m1^2)/(2*s1^2)
    - [0 0 1]*log(P(2)/P(1)*s(1)/s(2));          % ln(P2/P1*s1/s2);

% Classify each of the samples
% Class 2 if p(1)*Y^2 + p(2)*Y + p(3) < 0
class_meas = cellfun(@(Y) (p(1)*Y.^2 + p(2)*Y + p(3) < 0) + 1, ...
    Y, 'UniformOutput', false);

% Create confusion matrix
confusion_matrix = zeros(length(class_meas));
for i = 1:size(confusion_matrix,1)
    for j = 1:size(confusion_matrix,1)
        confusion_matrix(i,j) = sum(class_meas{i} == j);
    end
end

% Display confusion matrix
disp('Confusion Matrix:');
disp(confusion_matrix);

% Compute number of errors
num_errors = sum(confusion_matrix,2) - diag(confusion_matrix);

% Compute conditional error probabilities
Pe = num_errors./sum(confusion_matrix,2);

% Compute probability of error based on assumed class frequency
Pe = sum(Pe(:).*P(:));

% Display probability of error
disp('Probability of Error:');
disp(Pe);