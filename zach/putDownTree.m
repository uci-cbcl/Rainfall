function out= putDownTree(X, cutval, cutvar, children, muLR, currNode, dataIdx)
b = X(dataIdx,cutvar(currNode)) <= cutval(currNode);
dataL = find(b);    dataR = find(~b);
childL = children(1,currNode);  childR = children(2,currNode);
% if child is leaf, take the value, else put indices down tree recursively
out = nan(numel(dataIdx),1); 
if children(1,childL)==0 % left child is a leaf
    out(dataL) = muLR(1,currNode);
else
    out(dataL) = putDownTree(X, cutval, cutvar, children, muLR, childL, dataIdx(dataL));
end
if children(1,childR)==0 % right child is a leaf
    out(dataR) = muLR(2,currNode);
else
    out(dataR) = putDownTree(X, cutval, cutvar, children, muLR, childR, dataIdx(dataR));
end