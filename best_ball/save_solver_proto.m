function [] = save_solver_proto(prms,fileName,lr,snapPrefix)

testInterval = prms.maxIter;

fid = fopen(fileName,'w');
fprintf(fid,sprintf('net: "%s" \n',prms.path.netProtoFile));
fprintf(fid,sprintf('test_iter: %d \n',prms.testIter));
fprintf(fid,sprintf('test_interval: %d \n',testInterval));
fprintf(fid,sprintf('base_lr: %.8f \n',lr));
fprintf(fid,sprintf('lr_policy: "%s" \n',prms.learnPolicy));
if ~strcmp(prms.learnPolicy, 'fixed')
	fprintf(fid,sprintf('gamma: %.2f \n',prms.gamma));
end
if strcmp(prms.learnPolicy, 'step')
	fprintf(fid,sprintf('stepsize: %d \n',prms.stepSize));
end
fprintf(fid, sprintf('display: %d \n',prms.displayIter));
fprintf(fid, sprintf('max_iter: %d \n',prms.maxIter));
fprintf(fid, sprintf('momentum: %f \n',prms.momentum));
fprintf(fid, sprintf('weight_decay: %f \n',prms.weightDecay));
if prms.isSnapshot
	fprintf(fid, sprintf('snapshot: %d \n',prms.snapshotIter));
end
fprintf(fid, sprintf('snapshot_prefix: "%s" \n',snapPrefix));
fprintf(fid, sprintf('solver_mode: %s \n',prms.solverMode));
fprintf(fid, sprintf('device_id: %d \n',prms.deviceId));

fclose(fid);
end
