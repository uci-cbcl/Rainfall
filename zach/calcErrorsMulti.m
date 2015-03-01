% INPUTS
% p is probability of each class predicted by the model - Nxc
% y is Nx1 class labels in 1:c
% w is weights of each data point - default: ones
%  DOES NOT DO WEIGHTING ON ROC MEASURE
%OUTPUTS
% dev - deviance; -2*log likelihood
% confusionMat - if number classes == 2, this is rocArea, otherwise:
%   nClass x nClass matrix of class predictions - rows are
%   true labels, colomns are predicted; confMat(i,j) is how many true class
%   i's were predicted as j's.
% acc - accuracy with 0.5 split point
% mse - mean squared error
function [dev,acc,mse,confusionMat,fprDesired] = calcErrorsMulti(p,y,w,fprDesired)
[N,c] = size(p);
if size(y,2)==N
    y=y';
end
% y is a set of probabilities of class labels
if and(c~=1,and(min(y(:))>=0, max(y(:))<=1))
    yIsProb=true;
else
    yIsProb=false;
    y = y(:);
end

if nargin < 3 || isempty(w), w = ones(size(y)); end
if nargin < 4 || isempty(fprDesired), fprDesired = []; end
% make sure weights in same dimension as y
if yIsProb
    if size(w,2)==1
        w = reshape(repmat(w(:),1,c),size(y));
    end
else
    w = reshape(w,size(y));
end

if and(~yIsProb,any(or(y > c,y < 1)))
    error([sprintf('Lbls must be in [%i %i], labels given are: ',1,c) ...
            sprintf('%i ',unique(y))]);
end
% if p's are class labels, make predictions as probabilities: 1 for
% predicted class, 0 for others
%if isequal(unique(p),unique(y))
if c==1 && all(p(:)==floor(p(:)))
   p = full(sparse(1:N, double(p), ones(N,1))); 
% if values out of range, convert to probs by soft-max
elseif max(p(:))>1 || min(p(:))<0
    p = exp(p - mean(p,2)*ones(1,c));
    p = p ./ (sum(p,2)*ones(1,c));
end

p = double(p);  y = double(y);  w = double(w);
myEps = 1e-8;
if ~yIsProb
    % make the predictions in range [myEps 1-myEps] to avoid log zero
    %pDev = reshape(min(1-myEps,max(myEps,p(:))),size(p));
    pDev = min(1-myEps,max(myEps,p(:)));
    %ind2 = sub2ind(size(p),(1:N)',y);
    ind = (y(:)-1)*N+(1:N)';
    % indices of predictions of correct label
    dev = -2 * sum(w.*log(pDev(ind)))/sum(w(:));
else
    pDev = min(1-myEps,max(myEps,p));
    dev = -2 * sum(sum(w.*y.*log(pDev)))/sum(w(:));
end
% Calculate Accuracy
if nargout < 2,    return;  end
if yIsProb
    acc=[];
else
    [maxPred,maxInd] = max(p,[],2);
    acc = weightedMean(y == maxInd,w);
end
% Calculate Mean Squared Error
if nargout < 3,    return;  end
pMSE = p;
if yIsProb
    mse = sum(sum(w.*(y-pMSE).^2)) / c / sum(w(:));
else
    pMSE(ind) = 1 - pMSE(ind);
    mse = weightedMean(sum(pMSE.^2,2) / c, w);
end
% Calculate Confusion Matrix in case of multi-class, or RoC area for 2class
if nargout < 4,    return;  end

if yIsProb
   confusionMat=[];
   return;
end

if c==2
   %[FPR,TPR,rocArea] = calcFPRTPR(p(:,2),y-1);
   %confusionMat = rocArea;
   AP=calcPR(p(:,2),y(:)-1);
   confusionMat = AP;
   %{
   if ~isempty(fprDesired)
    fprDesired = zeros(size(fprDesired));
    for i=1:numel(fprDesired)
      tmp = TPR(find(FPR <= fprDesired(i), 1, 'last'));
      if isempty(tmp),  fprDesired(i) = nan;
      else,             fprDesired(i) = tmp;
      end
    end
   end
   %}
else
   confusionMat = zeros(c);
   for i=1:c
       predict = maxInd(y==i);
       for j=1:c
          confusionMat(i,j) =  sum(predict==j);
       end
   end
end

function m = weightedMean(x,w)
m = sum(w.*x) / sum(w);