function [acc, varargout] = log2accuracy(parseScript,fileName)
%Find the test accuracy from the log file

system(sprintf('%s %s',parseScript,fileName));
testFileName = strcat(fileName,'.test');

fid = fopen(testFileName,'r');
tline = '';
pLine = '';
while ischar(tline)
	pLine = tline;
	tline = fgetl(fid);
end
fclose(fid);

%The last line contains accuracy
pos = strfind(pLine,'  ');
acc = pLine(pos(2)+1:pos(3)-1);
acc = str2double(acc);

iters = pLine(1:pos(1)-1);
iters = str2num(iters);
varargout{1} = iters;

trainFileName = strcat(fileName,'.train');
system(['rm ' testFileName]);
system(['rm ' trainFileName]);

end
