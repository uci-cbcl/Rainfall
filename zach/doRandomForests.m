function [ predictY,mseTraining,mseValidation ] = ...
    doRandomForests( Xtrain,Xvalid,Ytrain,Yvalid,N,numRandFeatures)
%DOGRADIENTBOOSTING Summary of this function goes here
%   Detailed explanation goes here

[numData,~] = size(Xtrain);
[numTestData,~] = size(Xvalid);

dt = cell(1,N);
mseTraining = zeros(1,N);
mseValidation = zeros(1,N);

predictY = 0;
prediction = zeros(numTestData,N);

for k=1:N,
 
 [xb,yb] = bootstrapData(Xtrain,Ytrain,numData*0.8);
 dt{k} = treeRegress(xb,yb,'maxDepth',15,'minParent',8,'nFeatures',numRandFeatures);
 curY = predict(dt{k}, xb);
 
 %find training MSE at k
 mseTraining(k) = mean((curY-yb).^2);
 
 %find validation MSE
 prediction(:,k) = predict(dt{k}, Xvalid);
 predictY = mean(prediction(:,1:k),2);
 
 mseValidation(k) = mean((Yvalid-predictY).^2);
 
 k
 
end;

end

