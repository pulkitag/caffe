expName = 'normal_big';

%Get prms
prms = get_prms({'expName',expName});

%Get Paths
prms.path = get_path(prms);

baseLr   = 0.001;
numRound = 50;
prms.isResume = false;
for r=1:1:numRound
	try	
		%Select the best learning rate
		baseLr = get_best_lr(prms, baseLr, r);
		disp(sprintf('BestLr: %f',baseLr));
	catch myErr
		disp('Running round');	
		run_round(prms, baseLr, r);
	end	
	%Get the best caffe model
	prms.isResume = true;	
	resFile = sprintf(prms.path.snapshot,r, baseLr);
	numIter = r*prms.iterPerRound;
	resFile = sprintf('%s_iter_%d.solverstate',resFile,numIter);
	prms.path.resumeFile =  resFile;
end

