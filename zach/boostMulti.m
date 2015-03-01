% IMPLEMENTS MULTI CLASS LOGIT BOOST WITH ARBITRARY BASE LEARNER.
%
%
% funfitval takes in: X,XTest,z,w,wtrimind,prevfuns,args{:}
% and returns newfun,f,fTest, f,ftest are newfun evaluated on X,XTest
% args: struct
%       nIter: how many boost iter to do
%       v: "learn rate": 0.25 is good
%       funargs: to be passed to funfitval
%       evaliter: which iter to eval performance at
% OUTPUTS:
%  perfTest: structure holding performance on test set
%  perfTrain: structure holding performance on train set
%  casesTrained: nIter x K, number of patterns used in training each class
%    at each iter - should decrease due to weight trimming
%  boostStruct: structure holding a lot of stuff.  Can be used as input to
%  this function to start where we left off
function [perfTest,perfTrain,casesTrained,boostStruct] = ...
    boostMulti(X,Y,XTest,YTest,funfitval,args,wInit,boostPrev,doDisp)
% parse args
% train=test, dont copy XTest: too big!
if isempty(YTest)
    YTest = [];
end
Y = double(Y); YTest = double(YTest);
N=numel(Y);
NTest = numel(YTest);
if NTest==0
    NTest = N;
end
if ~exist('doDisp') || isempty(doDisp)
    doDisp=true;
end
if isempty(args)
    args.nIter = 1000;    args.v = 0.25;
    args.funargs = {};    args.evaliter = [1 10:10:args.nIter];
end

nIter = args.nIter;
v = args.v;
funargs = args.funargs;
evaliter = args.evaliter; % which iterations we are evaluating error at
if numel(unique(evaliter))~=numel(evaliter)
    error('eval iter contains same iteration >= 2 times');
end

%Y = single(Y); YTest = single(YTest);
% rename the y's as 1,2,...
[y,classNamesStr] = grp2idx(Y);
if isa(Y,'single')
    y = single(y);
end

% make class names integers, not strings
classNames = zeros(size(classNamesStr));
for i=1:numel(classNames)
    classNames(i) = str2double(classNamesStr{i});
end
% indicator response matrix
tmp = [y(:) y(:)]; tmp(:,1) = 1:N;
one = tmp(1); one(1)=1;
Y = accumarray(tmp,one);
clear tmp;
%YOLD = full(sparse(1:N, y, ones(N,1)));
[N,K] = size(Y);
% make yTest same numbering as y
yTest = double(YTest);
for i=1:numel(classNames)
   yTest(YTest == classNames(i)) = i;
end


if nargin < 4 || isempty(wInit)
    wInit = ones(size(Y,1),1);
end
assert(isequal(numel(wInit),N), 'bad wInit');
wInit = wInit(:);
%%%%%%%%%%%%%%%%%%%%%% MULTI CLASS LOGIT BOOST %%%%%%%%%%%%%%%%%%%%%%%%%%%%
if ~exist('boostPrev','var') || isempty(boostPrev)
    F = zeros(N,K);
    FTest = zeros(NTest,K);
    startIter = 1;
    casesTrained = zeros(nIter,K);
    startPerfInd = 0;
    oldEvalIter = [];
    perfTest = [];
    perfTrain = [];
    funparams=cell(nIter,K);
else
    F = boostPrev.F;
    FTest = boostPrev.FTest;
    startIter = boostPrev.startIter;
    casesTrained = boostPrev.casesTrained;
    perfTest = boostPrev.perfTest;
    perfTrain = boostPrev.perfTrain;
    oldEvalIter = boostPrev.oldEvalIter;
    funparams=boostPrev.funparams;
    startPerfInd = numel(oldEvalIter);
end

z = zeros(size(Y,1),1);
p = ones(size(Y)) / K;
f = zeros(size(Y));
fTest = zeros(NTest, K);
wMin = 1e-10; % threshold min weight
zMax = 4;
w = wInit;
initSumW = sum(w);
wTrim = 0.1;

%perfTrain = cell(numel(evaliter), 1);
%perfTest = cell(numel(evaliter), 1);
if doDisp,  clk=clock;    end
for i=startIter-1:nIter
   % skip this first time to initialize F,p below
   if i~=startIter-1
    for j=1:K
      %if j==2&&K==2
      %   f(:,2) = -f(:,1);
      %   fTest(:,2) = -fTest(:,1);
      %else
      %z(:,j) = (Y(:,j) - p(:,j)) ./ (p(:,j) .* (1-p(:,j)));
      y1Ind = Y(:,j) == 1;
      z(y1Ind) = min(zMax, 1 ./ (eps+p(y1Ind,j)));
      z(~y1Ind) = max(-zMax, -1 ./ (1-p(~y1Ind,j) + eps));
      w = p(:,j) .* (1 - p(:,j));
      w=w.*wInit;
      % perform weight trimming
      w = w * initSumW / sum(w); % norm weight sums
      w = max(w, wMin);
      wTrimInd = findWTrimInd(w, wTrim);
      %wTrimInd=[];
      casesTrained(i,j) = sum(wTrimInd);
      if isa(Y,'single')
         w = single(w); z=single(z);
      end
      %[f(:,j),fTest(:,j),currFun]=funfitval(X,XTest,z,w,wTrimInd,funparams(1:i-1,:),funargs{:});
      boostInfo.funparams = funparams;
      boostInfo.currIter = i;
      boostInfo.currClass= j;
      boostInfo.N = N;  boostInfo.NTest = NTest;
      [f(:,j),fTest(:,j),currFun]=funfitval(X,XTest,z,w,wTrimInd,boostInfo,funargs{:});
      funparams{i,j} = currFun;
      %end
    end
   end
   %if K==2
   %  f=1/2*f;   fTest=1/2*fTest; 
   %else
       f = (K-1)/K * (f - mean(f,2)*ones(1,K));
       fTest = (K-1)/K * (fTest - mean(fTest,2)*ones(1,K));
   %end
   F = F+v*f;
   FTest = FTest+v*fTest;
   p = exp(F);
   p = p ./ (sum(p,2)*ones(1,K));
   pTest = exp(FTest);
   pTest = pTest ./ (sum(pTest,2)*ones(1,K));
   boostStruct.p=p;
   boostStruct.pTest=pTest;
   tmp = find(i==evaliter);
   % don't calc perf on first iter: just here to initialize p,F
   if ~isempty(tmp) && i~=(startIter-1)
      tmp = tmp+startPerfInd; %?
      [perfTrain(end+1).dev, perfTrain(end+1).acc, perfTrain(end+1).mse, perfTrain(end+1).confMat] = ...
           calcErrorsMulti(p,y,wInit);
      oldEvalIter(end+1) = i;
      if doDisp
        fprintf('finished iter %i of %i  ', i, nIter);
      end

      if ~isempty(YTest)
        [perfTest(end+1).dev, perfTest(end+1).acc, perfTest(end+1).mse, perfTest(end+1).confMat] = ...
           calcErrorsMulti(pTest,yTest);

       if doDisp 
        if K~=2,
            fprintf('  testAcc = %-3.3g', perfTest(end).acc);
        else
            [AP,prec,recall]=calcPR(pTest(:,2),yTest(:)-1);
            fprintf('dev=%-3.3g testAcc=%-3.3g, Average Precision = %-3.3g',perfTest(end).dev,perfTest(end).acc,AP);
            %fprintf('  rocArea = %-3.3g', perfTest(end).confMat);
        end
       end
      end

      if doDisp,    fprintf('elapsed time %-3.3g seconds\n',etime(clock,clk));    end
       %save BoostEmergency.mat i F FTest p pTest;
   end
end

boostStruct.funparams = funparams;
boostStruct.F = F;
boostStruct.FTest = FTest;
boostStruct.startIter = i+1;
boostStruct.casesTrained = casesTrained;
boostStruct.perfTest = perfTest;
boostStruct.perfTrain = perfTrain;
boostStruct.oldEvalIter = oldEvalIter;
