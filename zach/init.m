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

Xtr = [X1tr X2tr];
Xte = [X1te X2te];

%%


%NOTE: cross-validation did not make a difference with training and
%       validation error
[Xtrain,Xvalid,Ytrain,Yvalid] = splitData(Xtr,Ytr,0.8);

%number of ensembles
N = 200;

mseTraining = zeros(1,N);
mseValidation = zeros(1,N);

%alpha values
alpha = 0.25*ones(1,N);
dt = cell(1,N);

predictY = 0;
curY = 0;

for k=1:N,
 
 grad = 2*(curY - Ytrain);
 dt{k} = treeRegress(Xtrain,grad,'maxDepth',3);
 curY = curY - alpha(k) * predict(dt{k}, Xtrain);
 
 %find training MSE at k
 mseTraining(k) = mean((curY-Ytrain).^2);
 
 %find validation MSE
 predictY = predictY - alpha(k)*predict(dt{k}, Xvalid);
 mseValidation(k) = mean((Yvalid-predictY).^2);
 
end;

%%
plot(mseTraining,'r-');
hold on
plot(mseValidation,'g--');
xlabel('Number of Learners in Ensemble');
ylabel('Mean Squared Error');
legend('Training Error','Validation Error');
title('MSE versus Number of Learners for Gradient Boosting');
%%

%train on all the test data

N=100; %new number of ensembles
curY=0;
predictY=0;
for k=1:N,
 
 %train the k-th decision tree
 grad = 2*(curY - Ytr);
 dt{k} = treeRegress(Xtr,grad,'maxDepth',3);
 curY = curY - alpha(k) * predict(dt{k}, Xtr);
 
 %boost current prediction using that k-th decision tree
 predictY = predictY - alpha(k)*predict(dt{k}, Xte);
 
end;

%%
fh = fopen('kagglePrediction.csv','w');  % open file for upload
fprintf(fh,'ID,Prediction\n');      % output header line
for i=1:length(predictY),
    fprintf(fh,'%d,%d\n',i,predictY(i));  % output each prediction
end;
fclose(fh);                         % close the file