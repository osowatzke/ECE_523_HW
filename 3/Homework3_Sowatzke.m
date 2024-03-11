%% Problem 1
% Data vectors
X1 = [ 1,  1;
      -1, -1]';

X2 = [ 1, -1;
      -1,  1]';

% Function to increase dimension of data
phi = @(X)([ones(1,size(X,2)); ...
            sqrt(2)*X(1,:); ...
            sqrt(2)*X(2,:); ...
            sqrt(2)*X(1,:).*X(2,:); ...
            X(1,:).^2; ...
            X(2,:).^2]);

% Apply function to samples of eah data class
X1 = phi(X1);
X2 = phi(X2);

% Solve for the linear discriminat function
[w,b] = svm(X1,X2);

% Display solution for problem 1
fprintf('Problem 1:\n');
fprintf('w\n')
disp(w);
fprintf('b = %g\n', b);
fprintf('\n');

%% Problem 2
% Data vectors
X1 = [1 1;
      2 2;
      2 0].';
X2 = [0 0;
      1 0;
      0 1].';

% Create figure
f = figure(1);
f.Position(3) = 420;
f.Position(4) = 400;
clf;
hold on;

% Plot points for each class
scatter(X1(1,:),X1(2,:),50,'X','LineWidth',1.5);
scatter(X2(1,:),X2(2,:),50,'O','LineWidth',1.5);

% Set plot bounds
xlim([-1 3]);
ylim([-1 3]);

% Plot the weight vector and margin
plot([-1, 2.5],[2.5 -1],'k','LineWidth',1.5);
plot([-1 3],[3 -1],'k--');
plot([-1 2],[2 -1],'k--');
axis square;

% Label margin
doublearrow([0, 0.5],[1 1.5]);
text(0,1.25,'M');

% Add legend
legend('Class \omega_1','Class \omega_2','Weight vector');
box on;

% Solve for equivalent linear discrimant function
[w,b] = svm(X1,X2);

% Display solution for problem 2
fprintf('Problem 1:\n');
fprintf('w\n')
disp(w);
fprintf('b = %g\n', b);
fprintf('\n');

% Function solves for a linear discriminant function using a support
% vector machine
function [w, b] = svm(X1, X2)
    X_hat = [X1, -X2];
    H = X_hat'*X_hat;
    f = -ones(1,size(X_hat,2));
    A = [ones(1,size(X1,2)), -ones(1,size(X2,2))];
    b = 0;
    a = quadprog(H,f,[],[],A,b,zeros(size(X_hat,2),1));
    w = X_hat*a;
    idx = find(a >= 1e-4, 1);
    if idx > size(X1,2)
        y = -1;
        x = X2(:,idx - size(X1,2));
    else
        y = 1;
        x = X1(:,idx);
    end
    b = y - w.'*x;
end

% Function converts a position on a plot to normalized units on the axis
function [x,y] = toNormalizedUnits(x,y)

    % Determine plot limits
    xLim = xlim;
    yLim = ylim;

    % Determine the span of the x and y axis
    xSpan = xLim(2) - xLim(1);
    ySpan = yLim(2) - yLim(1);

    % Compute a scale factor for mapping x and y coordinates to pixels
    ax = gca;
    xSf = (ax.Position(3)*ax.Parent.Position(3))/xSpan;
    ySf = (ax.Position(4)*ax.Parent.Position(4))/ySpan;

    % Take the minimum scaling for square axis
    sf = min([xSf,ySf]);

    % Determine scale factors for mapping x and y cordinates to normalized
    % units
    xSf = sf/ax.Parent.Position(3);
    ySf = sf/ax.Parent.Position(4);

    % Determin normalized position for start of xspan and start of yspan
    xCenter = ax.Position(1) + ax.Position(3)/2;
    yCenter = ax.Position(2) + ax.Position(4)/2;
    xStart = xCenter - xSpan/2*xSf;
    yStart = yCenter - ySpan/2*ySf;

    % Convert x-y position to normalized units
    x = (x - xLim(1))*xSf + xStart;
    y = (y - yLim(1))*ySf + yStart;
end

% Create a double arrow at x-y positions on the axis
function doublearrow(x,y)
    [x,y] = toNormalizedUnits(x,y);
    annotation('doublearrow',x,y)
end




