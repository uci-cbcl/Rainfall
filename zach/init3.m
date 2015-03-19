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
[Xtrain,Xvalid,Ytrain,Yvalid] = splitData(Xtr,Ytr,0.8);

%%
numRandFeats = 40;
numFeats = size(Xtrain,2);

N=50;
dt = cell(1,N);
mseTraining = zeros(1,N);
mseValidation = zeros(1,N);
featsUsed = zeros(N,numRandFeats);

for k=1:N,
    
    feats = randperm(numFeats);
    featsToUse = feats(1:numRandFeats);
    featsUsed(k,:) = featsToUse;
    
    curXtrain = zeros(size(Xtrain,1),numRandFeats);
    curXvalid = zeros(size(Xvalid,1),numRandFeats);
    for i=1:numRandFeats
       curXtrain(:,i) = Xtrain(:,featsToUse(i)); 
       curXvalid(:,i) = Xvalid(:,featsToUse(i)); 
    end
 
 dt{k} = treeRegress(curXtrain,Ytrain,'maxDepth',15,'minParent',512);
 curY = predict(dt{k}, curXtrain);
 
 %find training MSE at k
 mseTraining(k) = mean((curY-Ytrain).^2);
 
 %find validation MSE
 predictY = predict(dt{k}, curXvalid);
 
 mseValidation(k) = mean((Yvalid-predictY).^2);
 
 k
 
end;

plot(mseTraining,'r-');
hold on
plot(mseValidation,'g-');
xlabel('Learner Number');
ylabel('Mean Squared Error');
legend('Training MSE','Validation MSE');
%%
[mse,bestIndices] = sort(mseValidation);
numGoodFeatures = 30;
bestFeatSets = zeros(numGoodFeatures,numRandFeats);
for i = 1:numGoodFeatures
   bestFeatSets(i,:) = featsUsed(bestIndices(i),:); 
end

%%

predictY = 0;
prediction = zeros(size(Xvalid,1),numGoodFeatures);
trainMSE = zeros(1,numGoodFeatures);
validMSE = zeros(1,numGoodFeatures);
for k=1:numGoodFeatures,
    
    curXtrain = zeros(size(Xtrain,1),numRandFeats);
    curXvalid = zeros(size(Xvalid,1),numRandFeats);
    for i=1:numRandFeats
       curXtrain(:,i) = Xtrain(:,bestFeatSets(k,i)); 
       curXvalid(:,i) = Xvalid(:,bestFeatSets(k,i)); 
    end
 
 [curPredictY,mseTraining,mseValidation] = ...
    doGradientBoosting(curXtrain,curXvalid,Ytrain,Yvalid,50);

 %find training MSE at k
 trainMSE(k) = min(mseTraining);
 
 %find validation MSE
 prediction(:,k) = curPredictY;
 predictY = mean(prediction(:,1:k),2);
 
 validMSE(k) = mean((Yvalid-predictY).^2);
 
 k
 
end;

plot(trainMSE,'r-');
hold on
plot(validMSE,'g--');
xlabel('Number of Learners in Ensemble');
ylabel('Mean Squared Error');
legend('Training Error','Validation Error');

%%

%assembles the Kaggle Data
predictY = 0;
prediction = zeros(size(Xte,1),numGoodFeatures);
for k=1:numGoodFeatures,
    
    curXtrain = zeros(size(Xtr,1),numRandFeats);
    curXvalid = zeros(size(Xte,1),numRandFeats);
    for i=1:numRandFeats
       curXtrain(:,i) = Xtr(:,bestFeatSets(k,i)); 
       curXvalid(:,i) = Xte(:,bestFeatSets(k,i)); 
    end
 
 [curPredictY,mseTraining,mseValidation] = ...
    doGradientBoosting(curXtrain,curXvalid,Ytr,0,50);
 
 %find validation MSE
 prediction(:,k) = curPredictY;
 predictY = mean(prediction(:,1:k),2);
 
 k
 
end;

makeKagglePrediction(predictY);
