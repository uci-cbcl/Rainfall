function pOut = boostTreeVal(boostStruct, nIter, XTest, v)
trees = boostStruct.funparams;
[nIterTmp,K] = size(trees);
nIter = min(nIter,nIterTmp);
[N,p] = size(XTest);
F = zeros(N,K);
f = zeros(N,K);
assert(isa(XTest,'uint8'), 'only uint8 trees implemented for now');
for n=1:nIter
    for k=1:K
        f(:,k) = treeValFAST(XTest,trees{n,k});
    end
    f = (K-1)/K * (f - mean(f,2)*ones(1,K));
    F = F + f;
end
F = F*v;
pOut = exp(F);
pOut = pOut ./ (sum(pOut,2)*ones(1,K));
