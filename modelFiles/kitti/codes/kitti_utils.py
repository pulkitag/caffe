import numpy as np
import my_pycaffe as mp
import my_pycaffe_utils as mpu
import rot_utils as ru
import pdb
import os

def get_paths():
	dirName = '/data1/pulkitag/data_sets/kitti/'
	prms    = {}
	prms['odoPath']  = os.path.join(dirName, 'odometry')
	prms['poseFile'] = os.path.join(prms['odoPath'], 'dataset', 'poses', '%02d.txt')
	return prms


def read_poses(seqNum=0):
	'''
		Provides the pose wrt to frame 1 in the form of (deltaX, deltaY, deltaZ, thetaZ, thetaY, thetaX
	'''
	paths  = get_paths()
	psFile = paths['poseFile'] % seqNum

	fid     = open(psFile, 'r')
	lines   = fid.readlines()
	allVals = np.zeros((len(lines), 5)).astype(float)
	for (i,l) in enumerate(lines):
		vals      = [float(v) for v in l.split()]
		tfmMat    = np.array(vals).reshape((3,4))
		rotMat    = tfmMat[:,0:3]
		translate = tfmMat[:,3]
		thetaZ, thetaY, thetaX = 
	fid.close()
	return allVals
	
	
 

 
