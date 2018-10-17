function [bestEpsilon bestF1] = selectThreshold(yval, pval)
%SELECTTHRESHOLD Find the best threshold (epsilon) to use for selecting
%outliers
%   [bestEpsilon bestF1] = SELECTTHRESHOLD(yval, pval) finds the best
%   threshold to use for selecting outliers based on the results from a
%   validation set (pval) and the ground truth (yval).
%

bestEpsilon = 0;
bestF1 = 0;
F1 = 0;


stepsize = (max(pval) - min(pval)) / 1000;
for epsilon = min(pval):stepsize:max(pval)
    tp=0;
    fp=0;
    fn=0;
    
    predictions = (pval < epsilon);
    tp = sum((predictions == 1) & (yval == 1));
    fn = sum((predictions == 0) & (yval == 1));
    fp = sum((predictions == 1) & (yval == 0));

    precision = tp/(tp+fp);
    recall = tp/(tp+fn);
    F1 = 2*precision*recall/(precision+recall);

    if F1 > bestF1
       bestF1 = F1;
       bestEpsilon = epsilon;
    end
end

end
