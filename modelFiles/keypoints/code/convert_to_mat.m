function [] = convert_to_mat()
% COnvert h5File into mat File
h5File = '/data1/pulkitag/keypoints/h5/test_labels_exprigid_lbluniform20_imSz128.hdf5';
matFile = '/data1/pulkitag/keypoints/h5/test_labels_exprigid_lbluniform20_imSz128.mat';

indices = h5read(h5File, '/indices');
indices = reshape(indices,4,[]);

save(matFile, 'indices','-v7.3');

keyboard;

end
