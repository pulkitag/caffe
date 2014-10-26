function [acc] = log2accuracy(parseScript,fileName)
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

trainFileName = strcat(fileName,'.train');
system(['rm ' testFileName]);
system(['rm ' trainFileName]);

end
