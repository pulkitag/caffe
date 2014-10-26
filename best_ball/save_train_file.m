function [] = save_train_file(prms, logFile)

fileName = prms.path.trainFile;
fid = fopen(fileName,'w');
fprintf(fid, '#!/usr/bin/env sh \n');
fprintf(fid, sprintf('TOOLS=%s/build/tools \n',prms.path.caffe));
fprintf(fid, sprintf('LOG_FILE_NAME=%s \n \n', logFile));

trainCmd = 'GLOG_logtostderr=1 $TOOLS/caffe train %s 2>&1 | tee -a ${LOG_FILE_NAME}';
trainStr = sprintf('--solver=%s ',prms.path.solverProtoFile);
if prms.isResume
	trainStr = strcat(trainStr,sprintf('--snapshot=%s',prms.path.resumeFile));
end

trainCmd = sprintf(trainCmd, trainStr);
fprintf(fid,trainCmd);

fclose(fid);
%Give run permissions
system(['chmod u+x ' fileName]);
end

