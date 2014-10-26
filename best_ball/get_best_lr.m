function [bestLr] = get_best_lr(prms, baseLr, roundNum)

lrPower     = prms.lrPower;
scriptFile  = fullfile(prms.path.caffe,'tools','extra','parse_log.sh');

acc = zeros(length(lrPower),1);
for l=1:1:length(lrPower)
	lr      = baseLr*power(2,lrPower(l));
	logFile = sprintf(prms.path.logFile,roundNum,lr); 
	acc(l)  = log2accuracy(scriptFile, logFile);
end

%Get the best learning rate. 
[~,idx] = max(acc);
bestLr  = baseLr*power(2,lrPower(idx));

end
