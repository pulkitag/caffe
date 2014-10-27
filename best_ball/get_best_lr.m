function [bestLr, varargout] = get_best_lr(prms, baseLr, roundNum)

lrPower     = prms.lrPower;
scriptFile  = fullfile(prms.path.caffe,'tools','extra','parse_log.sh');

acc = zeros(length(lrPower),1);
for l=1:1:length(lrPower)
	lr      = baseLr*power(2,lrPower(l));
	logFile = sprintf(prms.path.logFile,roundNum,lr); 
	[acc(l),iterCount]  = log2accuracy(scriptFile, logFile);
end

%Get the best learning rate. 
[bestAcc,idx] = max(acc);

%Encourage larger learning rates. 
midIdx = round(length(lrPower)/2);
if idx < midIdx
	if acc(midIdx) + 0.01 > bestAcc
		bestAcc = acc(midIdx);
		idx     = midIdx;
	end
end

bestLr  = baseLr*power(2,lrPower(idx));
varargout{1} = idx;
varargout{2} = iterCount;
varargout{3} = bestAcc;
varargout{4} = acc;
end
