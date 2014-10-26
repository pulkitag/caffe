expName = 'normal_big';

%Get prms
prms = get_prms({'expName',expName});

%Get Paths
prms.path = get_path(prms);

baseLr   = 0.001;
numRound = 10;
prms.isResume = false;
for r=1:1:numRound
	%Run a round
	run_round(prms, baseLr, r);

	%Select the best learning rate
	baseLr = get_best_lr(prms, baseLr, r);
	
	%Get the best caffe model
	prms.isResume = true;	
	resFile = sprintf(prms.paths.snapshot,r, baseLr);
	numIter = r*prms.iterPerRound;
	resFile = sprintf('%s_iter_%d.caffemodel',resFile,numIter);
	prms.path.resumeFile =  resFile;
end

