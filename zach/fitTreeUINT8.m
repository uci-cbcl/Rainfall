% J is the number of leaf nodes in the tree
% <= goes to the left, > goes to the right
function tree = fitTreeUINT8(X,y,w,wTrim,candVarsParam,J)
assert(isa(X,'uint8'),isa(w,'double'));
wy=w.*y;
wwy = [w(:) wy(:)]';
wy2 = wy.*y;
% WEIGHT TRIMMING NOT IMPLEMENTED
f = [];
[N,p] = size(X);

candVars = treeCandVars(candVarsParam, size(X,2), true);
[cutval,ssx,muL,muR] = fitStumpUINT8_c(X,wwy,uint32(candVars),f);
[mnSSX,mnInd] = min(ssx);
mnSSX = mnSSX + sum(wy.*y);
cutvarTree = candVars(mnInd)+1;
cutvalTree = cutval(mnInd);
ssxTree = mnSSX;
muLRTree = [muL(mnInd); muR(mnInd)];
% children: left and right children index. leaves have 0
children = zeros(2,1);
parent = zeros(1);
%bCell{1} = true(N,1);
%fndCell = cell(1,J*10); % pre-allocation *10 is a large amount of extra
fndCell{1} = uint32((1:N)');
% check that the muL and muR are correct
%{
for i=1:numel(candVars)
    b = X(:,candVars(i)+1)<=cutval(i);
    fL = find(b);   fR = find(~b);
    muLCheck(i) = sum(wy(fL))./(sum(w(fL))+eps);
    muRCheck(i) = sum(wy(fR))./(sum(w(fR))+eps);
end
b = X(:,candVars(cutvarTree)+1)<=cutvalTree;
fL = find(b);   fR = find(~b);
muLCheck = sum(wy(fL))./(sum(w(fL))+eps);
muRCheck = sum(wy(fR))./(sum(w(fR))+eps);
%}
next = 2;
while sum(children(1,:)==0) < J
    % find best leaf to split
    fnd = find(0 == children(1,:));
    ssxParent = 0*fnd;
    goodInd = parent(fnd)~=0;
    ssxParent(goodInd) = ssxTree(parent(fnd(goodInd)));
    [~,bestLeaf] = min(ssxTree(fnd) - ssxParent);
    bestLeaf = fnd(bestLeaf);
    
    % SPLIT LEAF: we've already computed the best split for this node, now
    %  send data points down the correct branch and split each of those
    children(:,bestLeaf) = size(children,2) + [1; 2];
    children(:,end+(1:2)) = 0;  % 2 leaves
    parent(:,end+(1:2)) = bestLeaf;    % I am your parent
    b = X(fndCell{bestLeaf}, cutvarTree(bestLeaf)) <= cutvalTree(bestLeaf);
    fndCell{next} = fndCell{bestLeaf}(b);
    fndCell{next+1} = fndCell{bestLeaf}(~b);
    next = next+2;
    %b = X(:,cutvarTree(bestLeaf)) <= cutvalTree(bestLeaf);
    %bCell{end+1} = and(bCell{bestLeaf},b);
    %bCell{end+1} = and(bCell{bestLeaf},~b);
    % cases that went left and cases that went right
    %fLR = {uint32(find(bCell{end-1})-1) uint32(find(bCell{end})-1)};
    fLR = {fndCell{end-1}-1 fndCell{end}-1};
    for jj=1:2
        candVars = treeCandVars(candVarsParam, size(X,2), true);
        [cutval,ssx,muL,muR] = fitStumpUINT8_c(X,wwy,candVars,fLR{jj});
        [ssxTree(end+1),mnInd] = min(ssx);
        %ssxTree(end) = ssxTree(end) + sum(wy(fLR{jj}+1).*y(fLR{jj}+1));
        ssxTree(end) = ssxTree(end) + sum(wy2(fLR{jj}+1));
        cutvalTree(end+1) = cutval(mnInd);
        cutvarTree(end+1) = candVars(mnInd) + 1;
        muLRTree(:,end+1) = [muL(mnInd); muR(mnInd)];
        %bCurr = intersect(fLR{jj}+1,find(X(:,cutvarTree(end))>cutvalTree(end)));
        1;
    end
end
%leaves = children(1,:)==0;
%muLRTree(:,~leaves) = nan; % on muLR in internal nodes; prevent future errors by setting to nan
tree.children = children;
tree.muLR = muLRTree;
tree.cutval = cutvalTree;
tree.cutvar = cutvarTree;
tree.ssxTree = ssxTree;