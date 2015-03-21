import numpy as np
import my_pycaffe as mp
import my_pycaffe_utils as mpu
import matplotlib.pyplot as plt
import my_pycaffe_io as mpio
import scipy.misc as scm
import rot_utils as ru
import pdb
import os

def get_paths():
	dirName = '/data1/pulkitag/data_sets/kitti/'
	svDir   = '/data0/pulkitag/projRotate/kitti/'
	prms    = {}
	prms['odoPath']     = os.path.join(dirName, 'odometry')
	prms['poseFile']    = os.path.join(prms['odoPath'], 'dataset', 'poses', '%02d.txt')
	prms['leftImFile']  = os.path.join(prms['odoPath'], 'dataset', 'sequences', '%02d','image_2','%06d.png')
	prms['rightImFile'] = os.path.join(prms['odoPath'], 'dataset', 'sequences', '%02d','image_3','%06d.png')
	prms['lmdbDir']     = os.path.join(svDir, 'lmdb-store') 
	return prms


def get_lmdb_names(expName, setName='train'):
	paths   = get_paths()
	if not setName in ['train', 'test']:
		raise Exception('Invalid Set Name')
	imFile  = os.path.join(paths['lmdbDir'], 'images_%s_%s-lmdb' % (setName, expName)) 
	lbFile  = os.path.join(paths['lmdbDir'], 'labels_%s_%s-lmdb' % (setName, expName))
	return imFile, lbFile


def get_num_images():
	allNum = [4541, 1101, 4661, 801, 271, 2761, 1101, 1101, 4071, 1591, 1201]
	return allNum


def read_poses(seqNum=0):
	'''
		Provides the pose wrt to frame 1 in the form of (deltaX, deltaY, deltaZ, thetaZ, thetaY, thetaX
	'''
	if seqNum > 10 or seqNum < 0:
		raise Exception('Poses are only present for seqNum 0 to 10')

	paths  = get_paths()
	psFile = paths['poseFile'] % seqNum

	fid     = open(psFile, 'r')
	lines   = fid.readlines()
	allVals = np.zeros((len(lines), 3, 4)).astype(float)
	for (i,l) in enumerate(lines):
		vals      = [float(v) for v in l.split()]
		allVals[i]    = np.array(vals).reshape((3,4))
	fid.close()
	return allVals


def read_images(seqNum=0, cam='left', imSz=256):
	'''
		Read the required images
	'''
	if cam=='left':
		imStr = 'leftImFile'
	elif cam=='right':
		imStr = 'rightImFile'
	else:
		raise Exception('cam type not recognized')

	paths    = get_paths()
	fileName = paths[imStr] % (seqNum, 0)
	dirName  = os.path.dirname(fileName)	
	N        = len(os.listdir(dirName))
	
	ims = np.zeros((N, imSz, imSz, 3)).astype(np.uint8)
	for i in range(N):
		fileName = paths[imStr] % (seqNum, i)
		im       = plt.imread(fileName)
		im       = scm.imresize(im, (256, 256))
		ims[i]   = im
	return ims	


def get_image_pose(seqNum=0, cam='left', imSz=256):
	poses  = read_poses(seqNum)
	ims    = read_images(seqNum, cam, imSz)
	return ims, poses


def get_pose_stats(poseType):
	'''
		Compute the pose stats by sampling 100 examples from each sequence
	'''
	if poseType=='euler':
		lbLength = 6
	else: 
		raise Exception('Unrecognized poseType')

	allPose = np.zeros((100 * 11, lbLength))
	count = 0
	for seqNum in range(0,11):
		poses = read_poses(seqNum)
		N     = poses.shape[0]
		perm  = np.random.permutation(N-1)
		perm  = perm[0:100]
		for i in range(100):
			p1, p2 =	poses[perm[i]], poses[perm[i]+1]
			allPose[count]  = get_pose_label(p1, p2, poseType).reshape(lbLength,)
			count += 1
			
	muPose = np.mean(allPose,axis=0).reshape((lbLength,1,1))
	sdPose = np.std(allPose, axis=0).reshape((lbLength,1,1))

	maxSd       = np.max(sdPose)
	scaleFactor = sdPose / maxSd 
	return muPose, sdPose, scaleFactor


def make_consequent_lmdb(poseType='euler', imSz=256, nrmlz='zScoreScale'):
	'''
		Take left and right images from all the sequences, get the poses and make the lmdb.
	'''
	expName = 'consequent_pose-%s_nrmlz-%s_imSz%d' % (poseType, nrmlz, imSz) 
	imF, lbF = get_lmdb_names(expName, 'train')
	db       = mpio.DoubleDbSaver(imF, lbF)
	seqCount = get_num_images()
	totalN   = 2 * sum(seqCount) - 2 * len(seqCount)	#2 times for left and right images
	perm     = np.random.permutation(totalN)

	if poseType == 'euler':
		poseLength = 6
	else:
		raise Exception('Pose Type Not Recognized')	

	if nrmlz=='zScore':
		muPose, sdPose,_ = get_pose_stats(poseType)
	elif nrmlz=='zScoreScale':
		muPose, sdPose, scale = get_pose_stats(poseType)
	else:
		raise Exception('Nrmlz Type Not Recognized')

	count    = 0
	for seq in range(0,11):
		for cam in ['left', 'right']:
			ims, poses    = get_image_pose(seq, cam=cam, imSz=imSz)
			N, nr, nc, ch = ims.shape
			imBatch = np.zeros((N-1, 2*ch, nr, nc)) 
			lbBatch = np.zeros((N-1, poseLength, 1, 1))
			for i in range(0, N-1):
				imBatch[i,0:ch,:,:] = ims[i].transpose((2,0,1))
				imBatch[i,ch:,:,:]  = ims[i+1].transpose((2,0,1))
				pose1, pose2        = poses[i], poses[i+1]
				lbBatch[i] = get_pose_label(pose1, pose2, poseType)	
	
			if nrmlz == 'zScore':	
				lbBatch = lbBatch - muPose
				lbBatch = lbBatch / sdPose	
			elif nrmlz == 'zScoreScale':
				lbBatch = lbBatch - muPose
				lbBatch = lbBatch / sdPose	
				lbBatch = lbBatch * scale
			else:
				raise Exception('Nrmlz Type Not Recognized')
			db.add_batch((imBatch, lbBatch), svIdx=(perm[count:count+N-1],perm[count:count+N-1]))		
			count = count + N-1
	

def get_pose_label(pose1, pose2, poseType):
	'''
		Returns the pose label 
	'''
	t1 = pose1[:,3]
	t2 = pose2[:,3]
	r1 = pose1[:3,:3]
	r2 = pose2[:3,:3]
	if poseType == 'euler':
		lb = np.zeros((6,1,1))
		lb[0:3] = (t2 - t1).reshape((3,1,1))
		lb[3], lb[4], lb[5] = ru.mat2euler(np.dot(r2.transpose(), r1))
	else:
		raise Exception('Pose Type Not Recognized')	

	return lb		

