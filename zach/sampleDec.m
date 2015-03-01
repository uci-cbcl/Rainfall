feaToUse = 1:size(X1tr,2);
[XTrPct,XTePct,binLocs] = XToPct(X1tr(:,feaToUse),X1te(:,feaToUse), 256);
XTrPct = uint8(XTrPct); XTePct = uint8(XTePct);
rf = 1; J=8; % J is number of leaf nodes in tree, rf is "random forest" parameter - # random features to choose at each iter
boostArgs.nIter = 100;  boostArgs.evaliter = unique([1:10:boostArgs.nIter boostArgs.nIter]);
boostArgs.v = 0.1;     boostArgs.funargs = {rf J};
boostStruct=[];
[perfTest,perfTrain,boostStruct] = BoostLS(XTrPct, Ytr, XTePct, [], @boostTreeFun, boostArgs, [], boostStruct);