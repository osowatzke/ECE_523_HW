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
mu = cellfun(@(x) mean(x).', IRIS_data,'UniformOutput',false);
S = cellfun(@(x) cov(x), IRIS_data,'UniformOutput',false);
S = reshape(S,1,1,length(S));

% Compute Fisher discriminant
Sw = sum(cell2mat(S),3);
w = Sw^(-1)*(mu{1} - mu{2});

% Compute decision threshold
c = ((mu{1} + mu{2})/2);
t = w'*c;

% Project means onto discriminant
mu1 = w'*mu{1};
mu2 = w'*mu{2};

% Classify each of the samples
if (mu1 > mu2)
    class_meas = cellfun(@(A) (A*w < t) + 1, IRIS_data, 'UniformOutput', false);
else
    class_meas = cellfun(@(A) (A*w >= t) + 1, IRIS_data, 'UniformOutput', false);
end

% Create confusion matrix
confusion_matrix = zeros(length(class_meas));
for i = 1:size(confusion_matrix,1)
    for j = 1:size(confusion_matrix,1)
        confusion_matrix(i,j) = sum(class_meas{i} == j);
    end
end

% Display confusion matrix
disp(confusion_matrix);

% Plot projected data
figure(1)
clf;
hold on;
for i = 1:length(IRIS_data)
    histogram(IRIS_data{i}*w);
end

% Plot decision threshold
line(t*ones(1,2),ylim,'Color','black','LineStyle','--','LineWidth',1.5);
box on;

% Plot data, line to project onto, and threshold
if length(w) == 2

    % Create scatter plot of input data
    figure(2);
    clf;
    hold on;
    for i = 1:length(IRIS_data)
        scatter(IRIS_data{i}(:,1),IRIS_data{i}(:,2),'Filled');
    end

    % Compute best plot limits
    xbounds = xlim;
    ybounds = ylim;
    xrange = xbounds(2) - xbounds(1);
    yrange = ybounds(2) - ybounds(1);
    if (xrange > yrange)
        range_delta = xrange - yrange;
        ybounds(1) = ybounds(1) - range_delta/2;
        ybounds(2) = ybounds(2) + range_delta/2;
    else
        range_delta = yrange - xrange;
        xbounds(1) = xbounds(1) - range_delta/2;
        xbounds(2) = xbounds(2) + range_delta/2;
    end

    % Plot line to project onto
    m = w(2)/w(1);
    b = c(2) - m*c(1);
    x = xbounds;
    y = m*x+b;
    plot(x,y,'LineWidth',1.5);

    % Plot decision threshold
    m = -1/m;
    b = c(2)-m*c(1);
    y = m*x+b;
    plot(x,y,'LineWidth',1.5);

    % Set plot limits
    xlim([-2 8]);
    ylim([-5 5]);
    xlim(xbounds);
    ylim(ybounds);
    axis square;
    box on;
end