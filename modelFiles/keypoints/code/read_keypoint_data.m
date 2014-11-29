imSz = 128
fileName = '/data1/pulkitag/data_sets/keypoints/data_for_pulkit.mat';
outDir   = sprintf('/data1/pulkitag/keypoints/raw/imSz%d/', imSz);
classNames = {'aeroplane','bicycle','bird','boat','bottle',...
							'bus','car','cat','chair','cow',...
							'diningtable','dog','horse','motorbike','person'...
							'potted-plant','sheep','sofa','train','tvmonitor'};
data     = load(fileName);
numCl    = 20;

if ~(exist(outDir,'dir')==7)
	system(['mkdir -p ' outDir]);
end
for cl=1:1:20
	imCl = data.all_I{cl};
	ims  = zeros(size(imCl,1),imSz,imSz,3,'uint8');
	for i = 1:1:size(imCl,1)
		%BGR conversion just fo comatibility reason
		imRGB        = imresize(imCl{i},[imSz imSz]);
		imBGR        = imRGB(:,:,[3 2 1]);
		ims(i,:,:,:) = imBGR;
	end 
	view = permute(data.all_R{cl},[3 1 2]);
	clCount = size(imCl,1);

	saveFile = fullfile(outDir,sprintf('%s.mat',classNames{cl}));
	save(saveFile,'ims','view','clCount','-v7.3');

end
