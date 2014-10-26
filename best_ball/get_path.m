function [pth] = get_path(prms,varargin)

pth = {'caffe','/work4/pulkitag-code/pkgs/caffe-v2-2/',...
			'runDirPrefix','best_ball',...
			'snapshotDir','/data1/pulkitag/snapshots/',...
			'netProtoFile','train_test.prototxt'};
pth = get_defaults(varargin,pth,true);

expPath = fullfile(pth.caffe,'modelFiles',prms.dataSetName,...
						pth.runDirPrefix,prms.expName);

snpPath = fullfile(pth.snapshotDir,prms.dataSetName,...
						pth.runDirPrefix,	prms.expName);

pth.solverProtoFile = fullfile(expPath,'solver.prototxt');
pth.trainFile       = fullfile(expPath,'train.sh');
pth.logFile         = fullfile(expPath,'logs/round%d_lr%.8f.txt');
pth.snapshot        = fullfile(snpPath,'snap_round%d_lr%.8f');
pth.netProtoFile    = fullfile(expPath,pth.netProtoFile);

if ~(exist(expPath,'dir')==7)
	system(['mkdir -p ' expPath]);
	system(['mkdir -p ' fullfile(expPath,'logs')]);
end
if ~(exist(snpPath,'dir')==7)
	system(['mkdir -p ' snpPath]);
end
end
