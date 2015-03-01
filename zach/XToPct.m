function [XPct,XTestPct,pct] = XToPct(X,XTest,nBins)
if ~exist('XTest','var') || isempty(XTest)
    XTest = [];
end
if ~exist('nBins','var') || isempty(nBins)
    nBins = 256;
end
%assert(nBins<=256);
[N,p] = size(X);
pct = nan(nBins,p);
L = linspace(0,100,nBins);
for i=1:p
    %u = unique(X(:,i));
    u = X(:,i);
    pct(:,i) = prctile(u(:),L); pct(end,i)=inf;
end
%     pct = prctile(X,linspace(0,100,256),1);
XPct = zeros(size(X));%,'uint8');
for i=1:p
    [~,tmp] = histc(X(:,i),pct(:,i));
    XPct(:,i) = tmp(:);
end
if isempty(XTest)
    XTestPct = [];
else
    assert(size(XTest,2)==p);
    XTestPct = zeros(size(XTest));%,'uint8');
    for i=1:p
        [~,tmp] = histc(XTest(:,i),pct(:,i));
        XTestPct(:,i) = tmp(:);
    end
end
