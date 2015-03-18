function [] = read_keypoint_data()

imSz = 128;
fileName = '/data1/pulkitag/data_sets/keypoints/data_for_pulkit_feb_2015.mat';
imStr    = '/data1/pulkitag/PASCAL/VOC2007_2012/VOCdevkit/VOC2007_2012/JPEGImages/%s.jpg';
outDir   = sprintf('/data1/pulkitag/keypoints/raw/imSz%d/', imSz);

classNames = {'aeroplane','bicycle','bird','boat','bottle',...
							'bus','car','cat','chair','cow',...
							'diningtable','dog','horse','motorbike','person'...
							'potted-plant','sheep','sofa','train','tvmonitor'};
dat     = load(fileName);
numCl   = 20;
contextPad = 16;
minSz      = 50; %If the maximum dimension of the object is below minSz - then ignore the object

if ~(exist(outDir,'dir')==7)
	system(['mkdir -p ' outDir]);
end
for cl=1:1:20
	imNames = dat.img_names{cl};
	bbox    = dat.all_bbox{cl};
	N       = length(imNames);
	ims     = zeros(N,imSz,imSz,3,'uint8');
	idx     = false(N,1);
	for i = 1:1:N
		mirrorFlag = false;
		if ~isempty(strfind(imNames{i},'mirror'))
			imNames{i} = imNames{i}(1:end-7);
			mirrorFlag = true;
		end
		im = imread(sprintf(imStr,imNames{i}));
		if mirrorFlag
			[hSz, wSz,~] = size(im);
			im = im(:,wSz:-1:1,:);
			clear hSz wSz;
		end
		[hSz, wSz, nCh] = size(im);
		bb = bbox{i};
	
		%Context Pad
		xSt  = max(1, bb(1) - contextPad);
		ySt  = max(1, bb(2) - contextPad);
		xEn  = min(wSz, xSt + bb(3) + 2*contextPad);
		yEn  = min(hSz, ySt + bb(4) + 2*contextPad);
	
		%Decide if the box needs to be ignored or not. 
		maxDiff = max(yEn - ySt, xEn - xSt) - 2*contextPad;
		if maxDiff < minSz
			continue;
		end 
		idx(i) = true;

		%Squarify
		[xSt,xEn,ySt,yEn] = squarify(xSt, xEn, ySt, yEn, wSz, hSz);	
		imCrop            = im(ySt:yEn,xSt:xEn,:);
		
		if mirrorFlag
			%keyboard;
		end	

		%BGR conversion just for comatibility reason
		imRGB        = imresize(imCrop,[imSz imSz]);
		imBGR        = imRGB(:,:,[3 2 1]);
		ims(i,:,:,:) = imBGR;
	end 
	view = permute(dat.all_R{cl},[3 1 2]);
	ims  = ims(idx,:,:,:);
	view = view(idx,:,:,:);
	clCount = sum(idx);

	disp(sprintf('For class:%d, %d instances found', cl, clCount));
	saveFile = fullfile(outDir,sprintf('%s.mat',classNames{cl}));
	save(saveFile,'ims','view','clCount','-v7.3');

end
end

function [xSt,xEn,ySt,yEn] = squarify(xSt, xEn, ySt, yEn, xSz, ySz)

	yDiff = yEn - ySt;
	xDiff = xEn - xSt;
	diff  = floor(abs(yDiff - xDiff)/2);

	if yDiff > xDiff
		xSt = max(1,xSt - diff);
		xEn  = min(xSz, xEn + diff);
	else
		ySt = max(1, ySt - diff);
		yEn = min(ySz, yEn + diff);
	end	

end
