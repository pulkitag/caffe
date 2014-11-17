%expName = 'topo_top_075';
expName = 'topo_top_075';
%Get prms
prms = get_prms({'expName',expName});

%Get Paths
prms.path = get_path(prms);

baseLr   = 0.001;
numRound = 50;
for r=1:1:numRound
	try	
		% For using previous calculations
		[baseLr,~, iterCount, bestAcc, acc] = get_best_lr(prms, baseLr, r);
		accStr = '';
		for l=1:1:length(acc)
			accStr = sprintf('%s %.4f', accStr, acc(l));
		end
		disp(sprintf('BestLr: %.8f, iterCount: %d: Acc: %f, all: %s',baseLr, iterCount, bestAcc, accStr));
	catch myErr
		break;
	end
end

