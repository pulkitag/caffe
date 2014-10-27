expName = 'topo_top_025';
leveldbNum = 2;
deviceId   = 2;

%Get prms
prms = get_prms({'expName',expName,'deviceId',deviceId});

%Get Paths
prms.path = get_path(prms);

if leveldbNum > 1
	modify_leveldb_name(prms.path.netProtoFile,leveldbNum);	
end

baseLr   = 0.001;
numRound = 50;
prms.isResume = false;
initIterPerRound = prms.iterPerRound;
iterCount        = 0;
maxIterCount     = 20000;

%Find the old rounds
oldRounds = 0;
for r=1:1:numRound
	try	
		% For using previous calculations
		[baseLr, idx, iterCount, bestAcc] = get_best_lr(prms, baseLr, r);
		disp(sprintf('BestLr: %f, iterCount: %d: Acc: %f',baseLr, iterCount, bestAcc));
		oldRounds = r;
	catch myErr
		oldRounds = r;
		break;
	end
end
keyboard;
pFile = fopen(prms.path.progressFile,'w');
for r=oldRounds:1:numRound
	disp('Running round');	
	run_round(prms, baseLr, r, iterCount, maxIterCount);
	iterCount = iterCount + prms.iterPerRound;
	
	%Select the best learning rate
	[baseLr, idx, ~, bestAcc] = get_best_lr(prms, baseLr, r);
	fprintf(pFile, 'Iter: %d, Acc: %f, lr: %f \n',iterCount, bestAcc, ...
						baseLr);
	disp(sprintf('BestLr: %f',baseLr));
	if idx==1 || idx==5
		prms.iterPerRound = initIterPerRound;
	else
		prms.iterPerRound = 2 * prms.iterPerRound;
	end

	%Get the best caffe model
	prms.isResume = true;	
	resFile = sprintf(prms.path.snapshot,r, baseLr);
	resFile = sprintf('%s_iter_%d.solverstate',resFile,iterCount);
	prms.path.resumeFile =  resFile;

	if iterCount > maxIterCount
		disp('Exceeded maxIterCount');
		break;
	end
end
fclose(pFile);

