function [ YhatTrain,Ytest ] = doGradBoostNick( Xtrain,Xtest,Ytrain )
%DOGRADBOOSTNICK Summary of this function goes here
%   Detailed explanation goes here

feaToUse = 1:size(Xtrain,2);
[XTrPct,XTePct,binLocs] = XToPct(Xtrain(:,feaToUse),Xtest(:,feaToUse), 256);
XTrPct = uint8(XTrPct); XTePct = uint8(XTePct);
rf = 1; J=8; 
% J is number of leaf nodes in tree
%rf is "random forest" parameter - # random features to choose at each iter
boostArgs.nIter = 200;  
boostArgs.evaliter = unique([1:10:boostArgs.nIter boostArgs.nIter]);
boostArgs.v = 0.1;     
boostArgs.funargs = {rf J};
boostStruct=[];
[perfTest,perfTrain,boostStruct] = ...
    BoostLS(XTrPct, Ytrain, XTePct, [], @boostTreeFun, boostArgs, [], boostStruct);

YhatTrain = boostStruct.F;
Ytest = boostStruct.FTest;

end

