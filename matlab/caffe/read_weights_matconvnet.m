function [] = read_weights_matconvnet(inFile, outFile)

net = load(inFile);
if isfield(net, 'net')
	net = net.net;
end
N = length(net.layers);
weights = {};
biases  = {};
names   = {};
count   = 1;
for i =1:1:N
	layer = net.layers{i};
	if strcmp(layer.type, 'conv')
		names{count}   = sprintf('conv%d', count);
		weights{count} = single(gather(net.layers{i}.filters));
		biases{count}  = single(gather(net.layers{i}.biases)); 	
		count = count + 1;
	end
end

save(outFile,'weights','biases','names');

end
