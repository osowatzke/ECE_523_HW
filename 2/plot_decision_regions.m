% Samples from class 1
X1 = [0 0; 0 1].';

% Samples from class 2
X2 = [1 0; 0 0.5].';

% Plot bounds
xrange = [-0.5 1.5];
yrange = [-0.5 1.5];

% Create new figure
figure(1)
clf;

% Color decision regions
hold on;
a1 = fill([xrange(1) xrange(1) 0.5 0.5],[yrange(1) 0.25 0.25 yrange(1)],...
    'r','FaceAlpha',0.5);
fill([xrange(1) 0.75 xrange(end) xrange(1)],...
    [0.75 0.75 yrange(end) yrange(end)], 'r','FaceAlpha',0.5);
a2 = fill([xrange(1) 0.5 0.5 xrange(end) xrange(end) 0.75 xrange(1)],...
    [0.25 0.25 yrange(1) yrange(1) yrange(end) 0.75 0.75],...
    'b','FaceAlpha',0.5);

% plot samples of class 1
s1 = scatter(X1(1,:),X1(2,:),50,'ko','LineWidth',1);

% plot samples of class 2
hold on;
s2 = scatter(X2(1,:),X2(2,:),50,'kx','LineWidth',1);

% plot lines between points
plot(xrange,zeros(1,2),'k--');
plot(zeros(1,2),yrange,'k--');
plot(xrange,flip(yrange),'k--')

% plot boundaries
plot([0.5 0.5],yrange,'k--');
plot(xrange,[0.25 0.25],'k--');
plot(xrange,xrange,'k--');
plot(xrange,[0.75 0.75],'k--');
box on;

% Set plot bounds
xlim(xrange);
ylim(yrange);

% Add legend
legend([s1,s2,a1,a2],'Class 1 Training Samples',...
    'Class 2 Training Samples','Class 1 Decision Region',...
    'Class 2 Decision Region')

