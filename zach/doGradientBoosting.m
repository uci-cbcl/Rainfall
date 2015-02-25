function [ predictY,mseTraining,mseValidation ] = ...
    doGradientBoosting( Xtrain,Xvalid,Ytrain,Yvalid,N)
%DOGRADIENTBOOSTING Summary of this function goes here
%   Detailed explanation goes here

maxTreeDepth=3;

mseTraining = zeros(1,N);
mseValidation = zeros(1,N);

%alpha values
alpha = 0.25*ones(1,N);
dt = cell(1,N);

predictY = 0;
curY = 0;

for k=1:N,
 
 grad = 2*(curY - Ytrain);
 dt{k} = treeRegress(Xtrain,grad,'maxDepth',maxTreeDepth);
 curY = curY - alpha(k) * predict(dt{k}, Xtrain);
 
 %find training MSE at k
 mseTraining(k) = mean((curY-Ytrain).^2);
 
 %find validation MSE
 predictY = predictY - alpha(k)*predict(dt{k}, Xvalid);
 mseValidation(k) = mean((Yvalid-predictY).^2);
 
end;

end

