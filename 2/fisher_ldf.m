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

% Compute number of correct classifications
num_correct_class = diag(confusion_matrix);

% Compute conditional probabilities of correct classifications
Pc = num_correct_class./sum(confusion_matrix,2);

% Using assumed class frequencies, compute classification accurracy
% (probability of correct classification)
Pc = sum(Pc(:).*P(:));

% Display classification accuracy
disp('Classification Accurracy:');
fprintf("\t%.2f%%\n",Pc*100);

% Plot projected data
figure(1)
clf;
hold on;
for i = 1:length(IRIS_data)
    histogram(IRIS_data{i}*w,10);
end

% Compute roots of polynomial describining decision boundaries
r = roots(p);

% Ignore complex roots
r = real(r(abs(imag(r)) < 1e-8));

% Plot decision boundaries
for i = 1:length(r)
    line(r(i)*ones(1,2),ylim,'Color','Black','LineWidth',1.5,...
        'LineStyle','--');
end
box on;

% Label plot
legend_str = repmat({''},2+length(r),1);
legend_str{1} = 'Class 1';
legend_str{2} = 'Class 2';
legend_str{3} = 'Threshold';
legend(legend_str,'Location','best');