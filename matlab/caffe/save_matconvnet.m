%First use the python function to generate the .mat file
%See my_pycaffe_io
%Then use swap_weights.

sourceFile = '/home/carreira/imagenet-caffe-alex.mat'; 
targetFile = '/data1/pulkitag/others/joao/kitti_slowness_concatfc6_ctMrgn1-60000.mat';
outFile    = '/data1/pulkitag/others/joao/kitti_slowness_concatfc6_ctMrgn1-60000-new.mat';
swap_weights_matconvnet(sourceFile, targetFile, outFile);
