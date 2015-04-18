import my_pycaffe as mp
import os
import numpy as np
import pdb
import my_pycaffe_io as mpio

SRC_DIR = '/data1/pulkitag/others/joao/dummy_refs'
TGT_DIR = '/data1/pulkitag/others/joao/matconv_output'

def get_info(preTrainStr, numIter):
	if preTrainStr in ['kitti_conv5', 'kitti_fc6', 'kitti_conv4']:
		snapshotDir = ('/data1/pulkitag/projRotate/snapshots/kitti/los-cls-ind-bn22_mxDiff-7'
										'_pose-sigMotion_nrmlz-zScoreScaleSeperate_randcrp_concat-fc6_nTr-1000000/')
		defDir = ('/work4/pulkitag-code/pkgs/caffe-v2-2/modelFiles/kitti/base_files')
		if preTrainStr == 'kitti_fc6':
			modelName = 'caffenet_con-fc6_scratch_pad24_imS227_iter_%d.caffemodel' % numIter
			defFile = 'kitti_finetune_fc6_deploy_input.prototxt'
			outFile = 'kitti_fc6_%d.mat' % numIter
			inFile  = 'imagenet-caffe-alex.mat'
		elif preTrainStr == 'kitti_conv5':
			modelName = 'caffenet_con-conv5_scratch_pad24_imS227_con-conv_iter_%d.caffemodel' % numIter 
			defFile = 'kitti_finetune_conv5_deploy_input.prototxt'
			outFile = 'kitti_conv5_%d.mat' % numIter
			inFile  = 'imagenet-caffe-alex.mat'
		elif preTrainStr == 'kitti_conv4':
			modelName = 'caffenet_con-conv4_scratch_pad24_imS227_con-conv_mysimple_iter_%d.caffemodel' % numIter
			defFile = 'kitti_finetune_conv4_mysimple_deploy_input.prototxt'
			outFile   = 'kitti_conv4_mysimple_%d' % numIter
			inFile    = 'matconv_dummy_kitti_conv4_simple.mat'
	elif preTrainStr in ['pascal_rotlim30']:
		snapshotDir = '/data1/pulkitag/pascal3d/snapshots/pascal3d_imSz128_lbl-lim30-az6-el6_crp-contPad16_ns1e+05_mb50'
		defDir  = '/work4/pulkitag-code/pkgs/caffe-v2-2/modelFiles/pascal3d/base_files/'
		modelName = 'caffenet_scratch_unsup_fc6_iter_%d.caffemodel' % numIter 
		defFile = 'caffenet_deploy_fc6_input.prototxt'
		outFile = 'pascal_rotlim30_%d' % numIter
		inFile  = 'matconv_dummy_pascal128.mat'
	elif preTrainStr in ['pascal_uniform_fc6_drop', 'pascal_uniform_fc6', 'pascal_cls']:
		snapshotDir = '/data1/pulkitag/pascal3d/snapshots/pascal3d_imSz128_lbl-uni-az30el10_crp-contPad16_ns4e+04_mb50'		
		defDir  = '/work4/pulkitag-code/pkgs/caffe-v2-2/modelFiles/pascal3d/base_files/'
		defFile = 'caffenet_deploy_fc6_input.prototxt'
		inFile  = 'matconv_dummy_pascal128.mat'
		if preTrainStr == 'pascal_uniform_fc6_drop':
			modelName = 'caffenet_scratch_unsup_fc6_drop_iter_%d.caffemodel' % numIter
			outFile   = 'pascal_uniform_fc6_drop_%d.mat' % numIter
		elif preTrainStr == 'pascal_uniform_fc6':	
			modelName = 'caffenet_scratch_unsup_fc6_iter_%d.caffemodel' % numIter
			outFile   = 'pascal_uniform_fc6_%d.mat' % numIter
		elif preTrainStr == 'pascal_cls':
			modelName = 'caffenet_scratch_sup_noRot_fc6_iter_%d.caffemodel' % numIter	
			outFile   = 'pascal_cls_%d.mat' % numIter
	else:	
		raise Exception('%s is not valid preTrainStr' % preTrainStr)
	
	netFile = os.path.join(snapshotDir, modelName)
	defFile = os.path.join(defDir, defFile)
	outFile = os.path.join(TGT_DIR, outFile)
	inFile  = os.path.join(SRC_DIR, inFile)
	return defFile, netFile, inFile, outFile


def make_output(preTrainStr, numIter):
	defFile, netFile, inFile, outFile = get_info(preTrainStr, numIter)
	net = mp.MyNet(defFile, netFile)
	mpio.save_weights_for_matconvnet(net, outFile, matlabRefFile=inFile)		
	


