%%

%run this when loading on a new computer
%{
X1te = load('kaggle/kaggle.X1.test.txt');
X1tr = load('kaggle/kaggle.X1.train.txt');
X2te = load('kaggle/kaggle.X2.test.txt');
X2tr = load('kaggle/kaggle.X2.train.txt');
Ytr = load('kaggle/kaggle.Y.train.txt');
save('kaggleData.mat','X1te','Ytr','X1tr','X2tr','X2te');
%}
%%
%run this if above was already run on this computer
load('kaggleData.mat');

%%

%add features to X1 which are the mean and std of the X2 patches
meanX2tr = mean(X2tr,2);
stdX2tr = std(X2tr,0,2);
meanX2te = mean(X2te,2);
stdX2te = std(X2te,0,2);

Xtr = [X1tr meanX2tr stdX2tr];
Xte = [X1te meanX2te stdX2te];

%Xtr = [X1tr X2tr];
%Xte = [X1te X2te];

%%
%Call 2: Kaggle score was 0.64843, the best one
%dt = treeRegress(Xtr,Ytr,'maxDepth',20,'minParent',2^9);
[Xtrain,Xvalid,Ytrain,Yvalid] = splitData(Xtr,Ytr,0.8);

%see best value for maxDepth
maxDepthVals = [1 2 5 7 10 12 15 20 22 25 27 30 32 35 37 40 42 45 47 50 52 55 57 60];
numVals = size(maxDepthVals,2);
validMSE = zeros(1,numVals);
trainMSE = zeros(1,numVals);
for val = 1:numVals
    dt = treeRegress(Xtrain,Ytrain,'maxDepth',maxDepthVals(val),'minParent',2^9);
    trainMSE(val) = mse(dt,Xtrain,Ytrain);
    validMSE(val) = mse(dt,Xvalid,Yvalid);
end
%%
plot(maxDepthVals,validMSE,'g-');
hold on
plot(maxDepthVals,trainMSE,'r-');
xlabel('Max Depth Value');
ylabel('Mean Squared Error');
legend('Validation MSE','Training MSE','Location','NorthWest');
[minVal,minIndex] = min(validMSE);
bestDepthVal = maxDepthVals(minIndex);

%%

dt = treeRegress(Xtr,Ytr,'maxDepth',20,'minParent',2^9);
Yhat = predict(dt,Xte);
makeKagglePrediction(Yhat);

%%

Xtr = X1tr;
Xte = X1te;

%NOTE: cross-validation did not make a difference with training and
%       validation error
[~,Xvalid,~,Yvalid] = splitData(Xtr,Ytr,0.8);

N = 15;
numFeats = 50;

%{
validMSEs = zeros(1,6);
numFeatValues = [20 30 40 50 60 70];
for i = 1:6
    numFeats = numFeatValues(i);
    i
    [~,~,mseValidation] = doRandomForests(Xtr,Xvalid,Ytr,Yvalid,N,numFeats);
    validMSEs(i) = min(mseValidation);
end
plot(numFeatValues,validMSEs);
xlabel('Number of Feature Values');
ylabel('Validation MSE');
%}

[~,mseTraining,mseValidation] = doRandomForests(Xtr,Xvalid,Ytr,Yvalid,N,numFeats);


plot(mseTraining,'r-');
hold on
plot(mseValidation,'g--');
xlabel('Number of Learners in Ensemble');
ylabel('Mean Squared Error');
legend('Training Error','Validation Error');
title('MSE versus Number of Learners for Gradient Boosting');


%%

%now learn on rest of data
[predictY,~,~] = ...
    doRandomForests(Xtr,Xte,Ytr,0,N);
makeKagglePrediction(predictY);

%%
%NOTE: cross-validation did not make a difference with training and
%       validation error
[Xtrain,Xvalid,Ytrain,Yvalid] = splitData(Xtr,Ytr,0.8);

[~,mseTraining,mseValidation] = ...
    doGradientBoosting(Xtrain,Xvalid,Ytrain,Yvalid,150);


plot(mseTraining,'r-');
hold on
plot(mseValidation,'g--');
xlabel('Number of Learners in Ensemble');
ylabel('Mean Squared Error');
legend('Training Error','Validation Error');
title('MSE versus Number of Learners for Gradient Boosting');

%%

%train on all the test data
[predictY,~,~] = ...
    doGradientBoosting(Xtr,Xte,Ytr,0,100);

makeKagglePrediction(predictY);