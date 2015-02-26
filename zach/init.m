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

Xtr = X1tr;
Xte = X1te;

%NOTE: cross-validation did not make a difference with training and
%       validation error
[Xtrain,Xvalid,Ytrain,Yvalid] = splitData(Xtr,Ytr,0.8);

[numData,numFeatures] = size(Xtrain);
[numTestData,numFeats] = size(Xvalid);

N = 10;
dt = cell(1,N);
mseTraining = zeros(1,N);
mseValidation = zeros(1,N);

predictY = 0;
curY = 0;
prediction = zeros(numTestData,N);

numRandFeatures = 50;

for k=1:N,
 
 [xb,yb] = bootstrapData(Xtr,Ytr,numData);
 dt{k} = treeRegress(xb,yb,'maxDepth',15,'minParent',8,'nFeatures',numRandFeatures);
 curY = predict(dt{k}, xb);
 
 %find training MSE at k
 mseTraining(k) = mean((curY-yb).^2);
 
 %find validation MSE
 prediction(:,k) = predict(dt{k}, Xvalid);
 predictY = mean(prediction(:,1:k),2);
 
 mseValidation(k) = mean((Yvalid-predictY).^2);
 
end;

plot(mseTraining,'r-');
hold on
plot(mseValidation,'g--');
xlabel('Number of Learners in Ensemble');
ylabel('Mean Squared Error');
legend('Training Error','Validation Error');
title('MSE versus Number of Learners for Gradient Boosting');


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
[~,mseTraining,mseValidation] = ...
    doGradientBoosting(Xtr,Xte,Ytr,0,100);

makeKagglePrediction(predictY);