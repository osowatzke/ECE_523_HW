%% Problem 1
X1 = [ 1,  1;
      -1, -1]';

X2 = [ 1, -1;
      -1,  1]';

X = [X1 X2];

X = [ones(1,size(X,2));
     sqrt(2)*X(1,:);
     sqrt(2)*X(2,:);
     sqrt(2)*X(1,:).*X(2,:);
     X(1,:).^2;
     X(2,:).^2;
     ones(1,size(X,2))];

X(:,3:4) = -X(:,3:4);

H = X'*X;
f = -ones(size(X,2),1);
A = [1 1 -1 -1];
b = 0;
a = quadprog(H,f,[],[],A,b);
w = X*a;

% Problem 2
X1 = [1 1;
      2 2;
      2 0].';
X2 = [0 0;
      1 0;
      0 1].';
f = figure(1);
clf;
% f.Position(3) = min([f.Position(3), f.Position(4)]);
% f.Position(4) = f.Position(3);
scatter(X1(1,:),X1(2,:),50,'X','LineWidth',1.5);
hold on;
scatter(X2(1,:),X2(2,:),50,'O','LineWidth',1.5);
xlim([-1 3]);
ylim([-1 3]);
plot([-1, 2.5],[2.5 -1],'k','LineWidth',1.5);
plot([-1 3],[3 -1],'k--');
plot([-1 2],[2 -1],'k--');
axis square;
doublearrow([0, 0.5],[1 1.5]);
text(0,1.25,'M');
box on;
legend('Class \omega_1','Class \omega_2','Weight vector');

X1 = [1, 1;
      2, 2;
      2, 0]';
X2 = [0, 0;
      1, 0;
      0, 1]';
X = [X1 -X2];
H = X'*X;
f = -ones(size(X,2),1);
A = [1 1 1 -1 -1 -1];
b = 0;
options = optimoptions('quadprog','Display','iter');
[a,FVAL] = quadprog(H,f,[],[],A,b,zeros(6,1),[],[],options);
w = X*a;
[~,I] = max(a);
if (I <= 3)
    b = 1 - w'*X(:,I);
else
    b = -1 - w'*X(:,I);
end

function doublearrow(x,y)
    [x,y] = toNormalizedUnits(x,y);
    annotation('doublearrow',x,y)
end

function [x,y] = toNormalizedUnits(x,y)
    ax = gca;
    xLim = xlim;
    yLim = ylim;
    xRange = xLim(2) - xLim(1);
    yRange = yLim(2) - yLim(1);
    xSf = ax.Position(3)*ax.Parent.Position(3)/xRange;
    ySf = ax.Position(4)*ax.Parent.Position(4)/yRange;
    sf = min([xSf,ySf]);
    xSf = sf/ax.Parent.Position(3);
    ySf = sf/ax.Parent.Position(4);
    xCenter = ax.Position(1) + ax.Position(3)/2;
    yCenter = ax.Position(2) + ax.Position(4)/2;
    xStart = xCenter - xRange/2*xSf;
    yStart = yCenter - yRange/2*ySf;
    x = (x - xLim(1))*xSf + xStart;
    y = (y - yLim(1))*ySf + yStart;
end
