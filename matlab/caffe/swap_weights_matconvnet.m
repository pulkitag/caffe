function [] = swap_weights_for_matconvnet(sourceFile, targetFile, outFile)
%
% sourceFile: MatConvNet style file for import weights. 
% targetFile: .mat file containing the target weights which need to be
%             swapped into the sourceFile
% outFile: Where the output needs to be saved

srcData = load(sourceFile);
tgtData = load(targetFile);

N = length(srcData.layers)
layers        = cell(N,1);
classes       = srcData.classes;
normalization = srcData.normalization;   
count = 1;
for i=1:1:length(srcData.layers)
	lData = srcData.layers{i};
	wName = strcat(lData.name, '_w');
	bName = strcat(lData.name, '_b');
	if strcmp(lData.type,'conv') && ~isfield(tgtData,wName)
		disp(sprintf('Skipping: %s',wName));
		break;
	end
	layers{count} = srcData.layers{i};
	if strcmp(lData.type,'conv')
		wData = tgtData.(wName);
		bData = tgtData.(bName);
		disp(wName);
		disp(size(lData.filters));
		if strcmp(wName,'fc6_w')
			disp('HACKING fc6');
			layers{count}.filters = reshape(wData,6,6,256,2048);
		else
			layers{count}.filters = reshape(wData,size(lData.filters));
		end
		layers{count}.biases  = bData;
	end
	count = count + 1;
end
layers = layers(1:count-1);
save(outFile, 'layers', 'classes', 'normalization');

end
