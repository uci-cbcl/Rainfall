% for use with BoostMulti or BoostLS
function [f,fTest,tree] = boostTreeFun(X,XTest,z,w,wtrimind,prevfuns,randfor,J)
if ~exist('randfor','var')
    randfor = 0.01;
end
if ~exist('J','var')
    J=6;
end
isUINT8 = isa(X,'uint8');
if isUINT8
    tree = fitTreeUINT8(X,z,w,wtrimind,randfor,J);
else
    tree = treefit(X,z,'weights',w,'maxleaf',J,'splitmin',1,'randfor',randfor);
    tree.assignednode = []; % too much mem
end
if isUINT8
    f = treeValFAST(X,tree);
else
    f = treeval(tree,X);
end
%{
u = unique(f); mns = nan(1,numel(u)); %assert(numel(u)==J)
for i=1:numel(u)
    fnd = find(u(i)==f);
    mns(i) = sum(w(fnd).*z(fnd)) / sum(w(fnd));
end
%}
if isempty(XTest)
    fTest = f;
else
    if isUINT8
        fTest = treeValFAST(XTest,tree);
    else
        fTest = treeval(tree,XTest);
    end
end
1;