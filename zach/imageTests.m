%load the data once this file exists. create it with init.m
load('kaggleData.mat');

%%

[numData,numFeatures] = size(X2tr);

%get random number
randomPoint = floor(rand(1,1)*numData)+1;

%gets a random raw image
randomImage = X2tr(randomPoint,:);
curImage = reshape(randomImage,21,21);

%graph the image
imagesc(curImage);
colorbar;