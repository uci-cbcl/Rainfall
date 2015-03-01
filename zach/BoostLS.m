% funfitval takes in: X,XTest,z,w,wTrimInd,prevfuns,args{:}
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
function [perfTest,perfTrain,boostStruct] = ...
    BoostLS(X,Y,XTest,YTest,funfitval,args,wInit,boostPrev)
% parse args
[N,p] = size(X);
nIter = args.nIter;
v = args.v;
funargs = args.funargs;
evaliter = args.evaliter; % which iterations we are evaluating error at
if numel(unique(evaliter))~=numel(evaliter)
    error('eval iter contains same iteration >= 2 times');
end

if ~exist('wInit','var') || isempty(wInit)
    wInit = ones(size(Y,1),1);
end
if ~isequal(numel(wInit),N)
    error('bad wInit');
end
%%%%%%%%%%%%%%%%%%%%%% LS BOOSTING %%%%%%%%%%%%%%%%%%%%%%%%%%%%
tic;
[NTest] = size(XTest,1);
if ~exist('boostPrev','var') || isempty(boostPrev)
    F = zeros(N,1);
    FTest = zeros(size(XTest,1),1);
    startIter = 1;
    startPerfInd = 0;
    oldEvalIter = [];
    perfTest = [];
    perfTrain = [];
    funparams=cell(nIter,1);
else
    F = boostPrev.F;
    FTest = boostPrev.FTest;
    startIter = boostPrev.startIter;
    perfTest = boostPrev.perfTest;
    perfTrain = boostPrev.perfTrain;
    oldEvalIter = boostPrev.oldEvalIter;
    funparams=boostPrev.funparams;
    startPerfInd = numel(oldEvalIter);
end

f = zeros(size(Y));
fTest = zeros(NTest, 1);
tic;
nextIterCell={}; % stuff to pass to next iteration of base learner
for i=startIter-1:nIter
   if i~=startIter-1
       z = Y - F; % current residuals
       %[f(:,j),fTest(:,j),currFun,nextIterCell]=funfitval(X,Y,XTest,YTest,z,funparams(1:i-1,:),nextIterCell,funargs{:});
       prevFuns = []; %not done yet
       [f, fTest,currFun] = funfitval(X, XTest, z, wInit(:), [], prevFuns, funargs{:});
       funparams{i} = currFun;
   end
   F = F + v * f;
   FTest = FTest + v * fTest;
   % eval perf
   tmp = find(i==evaliter);
   if ~isempty(tmp)
      perfTrain(end+1).rmse = sqrt(sum((Y-F).^2.*wInit)/sum(wInit));
      oldEvalIter(end+1) = i;
      fprintf('finished iter %i of %i  ', i, nIter);
      if ~isempty(YTest)
        perfTest(end+1).rmse = sqrt(sum((YTest-FTest).^2) / N);
        fprintf('rmseTrain=%4.4f  rmseTest = %4.4f   ', perfTrain(end).rmse, perfTest(end).rmse);
      end
      toc; tic;
   end
end

boostStruct.funparams = funparams;
boostStruct.F = F;
boostStruct.FTest = FTest;
boostStruct.startIter = i+1;
boostStruct.perfTest = perfTest;
boostStruct.perfTrain = perfTrain;
boostStruct.oldEvalIter = oldEvalIter;
