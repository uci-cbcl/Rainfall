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

Xtr = X1tr;
[Xtrain,Xvalid,Ytrain,Yvalid] = splitData(Xtr,Ytr,0.8);

N=60;

%does a combination of random forest and boosting
[numData,numFeatures] = size(Xtrain);
[numTestData,~] = size(Xvalid);

mseTraining = zeros(1,N);
mseValidation = zeros(1,N);

predictY = 0;
prediction = zeros(numTestData,N);
numRandFeatures = 60;


curXtrain = zeros(numData,numRandFeatures);
curXvalid = zeros(numTestData,numRandFeatures);

for k=1:N,
    
    featNums = randperm(numFeatures);
    featNums = featNums(1:numRandFeatures);
    featNums = sort(featNums);
    for i = 1:numRandFeatures
       curXtrain(:,i) = Xtrain(:,featNums(i));
       curXvalid(:,i) = Xvalid(:,featNums(i));
    end
 
 [Yhat,Ytest] = doGradBoostNick(curXtrain,curXvalid,Ytrain);
 
 %find training MSE at k
 mseTraining(k) = mean((Yhat-Ytrain).^2);
 
 %find validation MSE
 prediction(:,k) = Ytest;
 predictY = mean(prediction(:,1:k),2);
 
 mseValidation(k) = mean((Yvalid-predictY).^2);
 
 k
 
end;


plot(mseTraining,'r-');
hold on
plot(mseValidation,'g--');
xlabel('Number of Learners in Ensemble');
ylabel('Mean Squared Error');
legend('Training Error','Validation Error');
title('MSE versus Number of Learners for Gradient Boosting');


%%

N=60;
Xtrain = X1tr;
Xvalid = X1te;
Ytrain = Ytr;

%does a combination of random forest and boosting
[numData,numFeatures] = size(Xtrain);
[numTestData,~] = size(Xvalid);

mseTraining = zeros(1,N);
mseValidation = zeros(1,N);

prediction = zeros(numTestData,N);
numRandFeatures = 60;


curXtrain = zeros(numData,numRandFeatures);
curXvalid = zeros(numTestData,numRandFeatures);

for k=1:N,
    
    featNums = randperm(numFeatures);
    featNums = featNums(1:numRandFeatures);
    featNums = sort(featNums);
    for i = 1:numRandFeatures
       curXtrain(:,i) = Xtrain(:,featNums(i));
       curXvalid(:,i) = Xvalid(:,featNums(i));
    end
 
 [Yhat,Ytest] = doGradBoostNick(curXtrain,curXvalid,Ytrain);
 
 %find validation MSE
 prediction(:,k) = Ytest;
 k
 
end;

predictY = mean(prediction,2);

makeKagglePrediction(predictY);