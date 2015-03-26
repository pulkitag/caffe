%Converts all the Pascal3D files into v7.3
pathName = '/data1/pulkitag/data_sets/pascal_3d/PASCAL3D+_release1.1/Annotations/';
outPath  = '/data1/pulkitag/data_sets/pascal_3d/PASCAL3D+_release1.1/Annotationsv73/';
classNames = {'aeroplane', 'bicycle', 'boat', 'bottle', 'bus', ...
							'car', 'chair', 'diningtable', 'motorbike', 'sofa',...
							'train', 'tvmonitor'};
datNames = {'pascal','imagenet'};

for c=1:1:length(classNames)
	for d=1:1:length(datNames)
		clName = sprintf('%s_%s', classNames{c}, datNames{d});
		classDir = fullfile(pathName, clName);
		outDir   = fullfile(outPath, clName);
		system(['mkdir -p ' outDir]);
		disp(classDir);
		fileNames = dir(sprintf('%s/*.mat',classDir));
		for l=1:1:length(fileNames)
			data = load(fullfile(classDir,fileNames(l).name));
			record = data.record;
			save(fullfile(outDir, fileNames(l).name), 'record', '-v7.3');	
		end
	end
end
