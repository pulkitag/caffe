function [] = modify_leveldb_name(fileName, num)

outFile = 'tmp.txt';
oFid    = fopen(outFile,'w');

leveldbPrefix = sprintf('-copy%d',num);

fid = fopen(fileName,'r');
fileLines = {};
tLine = '';
flag = false;
while (ischar(tLine))
	tLine = fgetl(fid);
	if ~isempty(strfind(tLine, 'data_param'))
		flag = true;
	end
	if flag==true && ~isempty(strfind(tLine, 'source: '))
		k = strfind(tLine,'"');
		sourceFile = tLine(k(1)+1:k(2)-1);
		sourceFile = strcat(sourceFile,leveldbPrefix);
		tLine      = sprintf('source: "%s"',sourceFile);	
	end
	if ischar(tLine)
		fprintf(oFid,tLine);
		fprintf(oFid,'\n');	
	end
end
fclose(fid);
fclose(oFid);

system(['mv ' outFile ' ' fileName]);
system(['rm ' outFile]);

end
