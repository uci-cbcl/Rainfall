% takes candVars  and returns absolute candVars
% efficient implementation.
function candVars = treeCandVars(candVars,p,toUINT32)
if numel(candVars)==1 && candVars<1
    if candVars>=0.25
        r=randperm(p); candVars=r(1:ceil(candVars*end));
    else
        %candVars=ceil(p*rand(1,ceil(p*candVars))); % candVars is small, so overlap will be little
        candVars = randi(p,[ceil(p*candVars) 1]);
    end
    %else
    %    nCand = ceil(p*candVars);    r = rand(1,p,'single');
    %    [candVars,candVars] = kMaxSingleFAST(r,nCand);
    %end
    candVars = sort(candVars);
elseif candVars==1
    candVars = 1:p;
end
if toUINT32
    candVars = uint32(candVars)-1;
end