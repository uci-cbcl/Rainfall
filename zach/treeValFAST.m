function f = treeValFAST(X,tree)
[N,p] = size(X);
try     muLR = tree.muLR;
catch   muLR = tree.class;
end
try     cutvar = tree.cutvar;
catch   cutvar = tree.var;
end
try     cutval = tree.cutval;
catch   cutval = tree.cut;
end
children = tree.children;
if size(children,1)~=2
    children = children';
end
f = putDownTree(X, cutval, cutvar, children, muLR, 1, (1:N)');
%{
f = zeros(N,1);
for n=1:N
   currNode = 1;
   while(children(1,currNode) ~= 0)
       idx = 1 + double(X(n,cutvar(currNode)) > cutval(currNode));
       f(n) = muLR(idx, currNode);
       currNode = children(idx, currNode);
   end
end
1;
%}


%{
for n=1:N
   currNode = 1;
   while(tree.children(1,currNode) ~= 0)
       if(X(n,tree.cutvar(currNode)) <= tree.cutval(currNode))
           f(n) = tree.muLR(1,currNode);
           currNode = tree.children(1,currNode);
       else
           f(n) = tree.muLR(2,currNode);
           currNode = tree.children(2,currNode);
       end
   end
end
1;
%}