function [  ] = makeKagglePrediction( predictY )
%MAKEKAGGLEPREDICTION Summary of this function goes here
%   Detailed explanation goes here

fh = fopen('kagglePrediction.csv','w');  % open file for upload
fprintf(fh,'ID,Prediction\n');      % output header line
for i=1:length(predictY),
    fprintf(fh,'%d,%d\n',i,predictY(i));  % output each prediction
end;
fclose(fh);                         % close the file

end

