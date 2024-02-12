% Load IRIS data
cell_data = readcell('IRIS_data.xlsx');

% Get class labels for data
label = cell_data(:,end);

% Optional select subset of features
% 2 features is good for visualization
cell_data = cell_data(:,1:2);

% Class names
class_names = {'setosa';'versicolor';'virginica'};

% Create empty cell arrays for training and test data
training_data = cell(length(class_names),1);
test_data = cell(length(class_names),1);

% Create empty cell arrays for distinguishing between different classes
class_training = cell(length(class_names),1);
class_test = cell(length(class_names),1);

% Populate cell arrays for each class
for i = 1:length(class_names)

    % Select sample corresponding to class
    data_sel = contains(label,class_names{i});
    class_data = cell2mat(cell_data(data_sel,:));

    % Determine the number of test and number of training samples
    num_class_samples = size(class_data,1);
    num_training_samples = floor(num_class_samples/2);
    num_test_samples = num_class_samples - num_training_samples;

    % Save data to cell arrays
    training_data{i} = class_data(1:num_training_samples,:);
    class_training{i} = repmat(i,num_training_samples,1);
    test_data{i} = class_data((num_training_samples+1):end,:);
    class_test{i} = repmat(i,num_test_samples,1);
end

% Convert cell arrays into array
training_data = cell2mat(training_data).';
class_training = cell2mat(class_training);
test_data = cell2mat(test_data);
class_test = cell2mat(class_test);
test_data = permute(test_data,[2,3,1]);

% For each test sample, get the index of the closet training sample
[~,I] = min(sum((test_data - training_data).^2));

% Determine the class of the closest training sample
class_meas = class_training(I);

% Populate confusion matrix
confusion_matrix = zeros(length(class_names));
for i = 1:size(confusion_matrix,1)
    confusion_matrix(:,i) = ...
        sum(class_meas(class_test == i) == (1:size(confusion_matrix,2)));
end
disp(confusion_matrix)

% If using two features show the training data, test data, and
% misclassified test samples
if size(test_data,1) == 2
    figure(1);
    clf;
    hold on;
    args = {'filled','o','MarkerFaceColor','blue';
        'filled','o','MarkerFaceColor','cyan';
        'filled','o','MarkerFaceColor','green'};
    for i = 1:length(class_names)
        plot_data = training_data(:,class_training == i);
        plot_args = args(i,:);
        scatter(plot_data(1,:),plot_data(2,:),plot_args{:});
    end
    
    args = {'x','MarkerEdgeColor','blue';
        'x','MarkerEdgeColor','cyan';
        'x','MarkerEdgeColor','green'};
    for i = 1:length(class_names)
        plot_data = test_data(:,class_meas == i);
        plot_args = args(i,:);
        scatter(plot_data(1,:),plot_data(2,:),plot_args{:});
    end
    err = class_meas ~= class_test;
    plot_data = test_data(:,err);
    scatter(plot_data(1,:),plot_data(2,:),'ro');
    box on;
    legend_str = [strcat(class_names,' (training)');
        strcat(class_names,' (predicted)'); {'error'}];
    legend(legend_str);
end