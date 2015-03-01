% INPUTS
% p1 is probability of class 1 predicted by model
% y is true 0-1 class labels of same size as predProb
% w is weights of each data point - default: ones
%  DOES NOT DO WEIGHTING ON ROC MEASURE
%OUTPUTS
% dev - deviance; -2*log likelihood
% roc - area under ROC curve
% acc - accuracy with 0.5 split point
% mse - mean squared error
function [dev,acc,mse,roc] = calcErrors(p1,y,w)
y = y(:);
N = numel(y);

% make p1 an N x z matrix where z is number of p1 predictions given
if size(p1,2) == N
    p1 = p1';
end
if size(p1,1) ~= N
    error(sprintf('N = %i, size p1 = [%i %i]',N,size(p1)));
end
% if only a single p1 given, do the same for all measures
if size(p1,2) == 1
    p1 = p1 * ones(1,4);
end
if size(p1,2) ~= 4
    error(sprintf('num p1 predictions = %i',size(p1,2)));
end

if nargin < 3, w = ones(size(y)); end
% make sure weights in same dimension as y
w = reshape(w,size(y));

if numel(union(unique(y),[0;1])) > 2
    error(sprintf('Lbls = %i  should  be [0 1]',unique(y)));
end

p1 = double(p1); y = double(y); w = double(w);
myEps = 1e-4;
% make the predictions in range [myEps 1-myEps] to avoid log zero
p1 = max(p1,myEps); p1 = min(p1,1-myEps);

devP1 = p1(:,1);
ly = logical(y);
dev = zeros(size(y));
dev(ly) = log(devP1(ly));
dev(~ly) = log(1-devP1(~ly));
dev = -2*weightedMean(dev,w);

accP1 = p1(:,2);
acc = weightedMean(y == (accP1 > 0.5),w);

mseP1 = p1(:,3);
mse = weightedMean((y - mseP1).^2,w);

% ROC AREA
if nargout < 4
    return;
end
warning('ROC AREA from CALCERRORS may be incorrect!!!');
rocP1 = p1(:,4);
[rocP1,ind] = sort(rocP1,'descend');
y = y(ind);
true1 = y == 1;
% number of true predicts 1's and model 1's that agree at each split
%nOnes = cumsum(true1);
%nModelOnes = 1:numel(y);

AB = sum(true1);
A = cumsum(true1);
CD = numel(y) - AB;
C = cumsum(~true1);

OneMinusSpe = C / CD;
Sen = A / AB;
[OneMinusSpe,ind] = sort(OneMinusSpe,'ascend');
Sen = Sen(ind);
roc = trapIntegrate(OneMinusSpe,Sen);
% plot(OneMinusSpe,Sen,'-o');

% integrate (x,y) points using trapezoidal rule
function T = trapIntegrate(x,y)
T = sum(diff(x).*(y(1:end-1)+y(2:end))/2);

function m = weightedMean(x,w)
m = sum(w.*x) / sum(w);