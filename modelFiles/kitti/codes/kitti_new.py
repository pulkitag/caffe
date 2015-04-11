import numpy as np
import kitti_utils as ku
import my_pycaffe_io as mpio
import other_utils as ou
import pdb
import os
import my_pycaffe_utils as mpu
import scipy.misc as scm
import myutils as myu
import copy

SET_NAMES = ['train', 'test']

##
# Resize and convert images to 256 by 256 for saving them.
def resize_images(prms):
	seqNum = range(11)
	rawStr = ['rawLeftImFile', 'rawRightImFile']
	imStr  = ['leftImFile', 'rightImFile']
	num    = ku.get_num_images()
	for raw, new in zip(rawStr, imStr):
		for seq in seqNum:
			N = num[seq]
			print seq, N, raw, new
			rawNames = [prms['paths'][raw] % (seq,i) for i in range(N)]			 
			newNames = [prms['paths'][new] % (seq,i) for i in range(N)]
			dirName = os.path.dirname(newNames[0])
			if not os.path.exists(dirName):
				os.makedirs(dirName)
			for rawIm, newIm in zip(rawNames, newNames):
				im = scm.imread(rawIm)
				im = scm.imresize(im, [256, 256])	
				scm.imsave(newIm, im)

##
# Save images as jpgs. 
def save_as_jpg(prms):
	seqNum = range(11)
	rawStr = ['rawLeftImFile', 'rawRightImFile']
	imStr  = ['leftImFile', 'rightImFile']
	num    = ku.get_num_images()
	for raw, new in zip(rawStr, imStr):
		for seq in seqNum:
			N = num[seq]
			print seq, N, raw, new
			rawNames = [prms['paths'][raw] % (seq,i) for i in range(N)]			 
			newNames = [prms['paths'][new] % (seq,i) for i in range(N)]
			dirName = os.path.dirname(newNames[0])
			if not os.path.exists(dirName):
				os.makedirs(dirName)
			for rawIm, newIm in zip(rawNames, newNames):
				im = scm.imread(rawIm)
				scm.imsave(newIm, im)

##
# Get the names of images
def get_imnames(prms, seqNum, camStr):
	N = ku.get_num_images()[seqNum]
	fileNames = [prms['paths'][camStr] % (seqNum,i) for i in range(N)]
	#Strip the imnames to only include last two folders
	imNames = []
	imSz    = []
	for f in fileNames:
		im   = ou.read_image(f)
		imSz.append(im.shape) 
		data = f.split('/')
		data = data[-3:]
		imNames.append((''.join(s + '/' for s in data))[0:-1])
	return imNames, imSz	

##
# Get normalized pose labels
def get_pose_label_normalized(prms, pose1, pose2):
	lbBatch = ku.get_pose_label(pose1, pose2, prms['pose'])
	muPose, sdPose = prms['poseStats']['mu'], prms['poseStats']['sd']
	scale          = prms['poseStats']['scale']

	if prms['nrmlz'] == 'zScore':	
		lbBatch = lbBatch - muPose
		lbBatch = lbBatch / sdPose	
	elif prms['nrmlz']	 == 'zScoreScale':
		'''
			This is good because if a variable doesnot 
			really changes, then there is going to 
			negligible change in image because of that. 
			So its not a good idea to just re-scale to
			the same scale on which other more important 
			factors are changing. So first make everything 
			sd = 1 and then scale accordingly. 
		'''
		lbBatch = lbBatch - muPose
		lbBatch = lbBatch / sdPose	
		lbBatch = lbBatch * scale
	elif prms['nrmlz'] == 'zScoreScaleSeperate':
		'''
			Same as zScorScale but scale the rotation and translations
			seperately. 
		'''
		nT = prms['numTrans'] #Number of translation dimensions.
		nR = prms['numRot']   #Number of rotation dimensions.
		transMax = np.max(scale[0:nT])
		rotMax   = np.max(scale[nT:])
		transScale = scale[0:nT] / transMax
		rotScale   = scale[nT:]  / rotMax
		scale      = np.concatenate((transScale, rotScale), axis=0)

		lbBatch = lbBatch - muPose
		lbBatch = lbBatch / sdPose	
		lbBatch = lbBatch * scale
	else:
		raise Exception('Nrmlz Type Not Recognized')

	if prms['lossType'] == 'classify':
		for i in range(lbBatch.shape[0]):
			#+1 because we clip everything below a certain value to the zeroth bin.
			lbBatch[i] = 1 + myu.find_bin(lbBatch[i].squeeze(), prms['binRange']) 

	return lbBatch
	
##
# Helper function for get_imnames and pose
def get_camera_imnames_and_pose(prms, seqNum, camStr, numSamples, randState=None):
	'''
		camStr    : The camera to use.
		seqNum    : The sequence to use.
		camStr    : left or right camera
		numSamples: The number of samples to extract.  
	'''
	if randState is None:
		randState = np.random

	#Get all imnames
	print "Reading images"
	imNames,imSz = get_imnames(prms, seqNum, camStr)

	print "Reading poses"
	poses     = ku.read_poses(prms, seqNum) 
	N         = len(imNames)
	mxFrmDiff = prms['maxFrameDiff']
	sampleIdx = randState.choice(N, numSamples) 

	im1, im2     = [], []
	imSz1, imSz2 = [], [] 
	psLbl    = []
	for i in range(numSamples):
		idx1 = sampleIdx[i]
		diff = int(round(randState.rand() * (mxFrmDiff)))
		sgnR = randState.rand()
		if sgnR > 0.5:
			diff = -diff
		idx2 = max(0, min(idx1 + diff, N-1))	
		#Add the images
		im1.append(imNames[idx1])
		im2.append(imNames[idx2])
		imSz1.append(imSz[idx1])
		imSz2.append(imSz[idx2])
		#Get the labels
		ps1, ps2 = poses[idx1], poses[idx2]
		psLbl.append(get_pose_label_normalized(prms, ps1, ps2))	
	return im1, im2, imSz1, imSz2, psLbl

##
# For a sequence, return list of imnames and pose-labels
def get_imnames_and_pose(prms, seqNum, numSamples, randState=None):
	camStrs = ['leftImFile', 'rightImFile']	
	im1, im2, psLbl = [], [], []
	imSz1, imSz2    = [], []
	numSamples = int(numSamples/2)
	for cam in camStrs:
		print 'Loading data for %s' % cam
		im11, im21, imSz11, imSz21,  psLbl1 = get_camera_imnames_and_pose(prms, seqNum, cam,
																					 numSamples, randState=randState)
		im1   = im1 + im11
		im2   = im2 + im21
		imSz1 = imSz1 + imSz11
		imSz2 = imSz2 + imSz21
		psLbl = psLbl + psLbl1
	return im1, im2, imSz1, imSz2, psLbl 


##
# Make the window file.
def make_window_file(prms):
	oldState  =  np.random.get_state()
	seqCounts =  ku.get_num_images()
	for sNum, setName in enumerate(['test', 'train']):
		seqNums     = ku.get_train_test_seqnum(setName)
		setSeqCount = np.array([seqCounts[se] for se in seqNums]).astype(np.float32)
		sampleProb  = setSeqCount / sum(setSeqCount) 

		im1, im2, ps = [], [], []
		imSz1, imSz2 = [], []
		for ii,seq in enumerate(seqNums):
			randSeed   = (101 * sNum) + 2 * seq + 1
			numSamples = int(round(prms['numSamples'][setName] * sampleProb[ii]))
			print "Storing %d samples for %d seq in set %s" % (numSamples, seq, setName) 
			randState = np.random.RandomState(randSeed)  
			imT1, imT2, imSzT1, imSzT2, psT = get_imnames_and_pose(prms,seq, numSamples, randState)
			im1   = im1 + imT1
			im2   = im2 + imT2
			imSz1 = imSz1 + imSzT1
			imSz2 = imSz2 + imSzT2 
			ps    = ps  + psT
	
		#Permute all the sequences togther
		perm         = randState.permutation(int(prms['numSamples'][setName]))
		im1          = [im1[p] for p in perm]
		im2          = [im2[p] for p in perm]
		imSz1        = [imSz1[p] for p in perm]
		imSz2        = [imSz2[p] for p in perm]
		ps           = [ps[p] for p in perm]
	
		#Save in the file
		gen = mpio.GenericWindowWriter(prms['paths']['windowFile'][setName],
						len(im1), 2, prms['labelSz'])
		for i in range(len(im1)):
			h,w,ch = imSz1[i]
			l1 = [im1[i], [ch, h, w], [0, 0, w, h]]
			h,w,ch = imSz2[i]
			l2 = [im2[i], [ch, h, w], [0, 0, w, h]]
			gen.write(ps[i], l1, l2)

	gen.close()
	np.random.set_state(oldState)


def get_solver(cPrms):
	solArgs = {'test_iter': 100,	'test_interval': 1000, 'base_lr': 0.001,
						 'gamma': 0.5, 'stepsize': 20000, 'max_iter': 250000,
				    'snapshot': 10000, 'lr_policy': '"step"', 'debug_info': 'true',
						'weight_decay': 0.0005}
	sol = mpu.make_solver(**solArgs) 
	return sol



def get_caffe_prms(concatLayer='fc6', concatDrop=False, isScratch=True, deviceId=1, 
									 contextPad=24, imSz=227): 
	caffePrms = {}
	caffePrms['concatLayer'] = concatLayer
	caffePrms['deviceId']    = deviceId
	caffePrms['contextPad']  = contextPad
	caffePrms['imSz']        = imSz

	expStr = []
	expStr.append('con-%s' % concatLayer)
	if isScratch:
		expStr.append('scratch')
	if concatDrop:
		expStr.append('con-drop')
	expStr.append('pad%d' % contextPad)
	expStr.append('imS%d' % imSz)	

	expStr = ''.join(s + '_' for s in expStr)
	expStr = expStr[0:-1]
	caffePrms['expStr'] = expStr
	caffePrms['solver'] = get_solver(caffePrms)
	return caffePrms

def get_experiment_object(prms, cPrms):
	caffeExp = mpu.CaffeExperiment(prms['expName'], cPrms['expStr'], 
							prms['paths']['expDir'], prms['paths']['snapDir'],
						  deviceId=cPrms['deviceId'])
	return caffeExp


def setup_experiment(prms, cPrms):
	#The size of the labels
	if prms['pose'] == 'euler':
		rotSz = 3
		trnSz = 3
	elif prms['pose'] == 'sigMotion':
		rotSz = 1
		trnSz = 2
	else:
		raise Exception('Unrecognized %s pose type' % prms['pose'])

	#The base file to start with
	baseFilePath = '/work4/pulkitag-code/pkgs/caffe-v2-2/modelFiles/kitti/base_files'
	baseFileStr  = 'kitti_siamese_window_%s' % cPrms['concatLayer']
	if prms['lossType'] == 'classify':
		baseStr = '_cls-trn%d-rot%d' % (trnSz, rotSz)
	else:
		baseStr = ''
	baseFile = os.path.join(baseFilePath, baseFileStr + baseStr + '.prototxt')
	print baseFile

	protoDef = mpu.ProtoDef(baseFile)	 
	solDef   = cPrms['solver']
	
	caffeExp = get_experiment_object(prms, cPrms)
	caffeExp.init_from_external(solDef, protoDef)

	#Get the source file for the train and test layers
	caffeExp.set_layer_property('window_data', ['generic_window_data_param', 'source'],
			'"%s"' % prms['paths']['windowFile']['train'], phase='TRAIN')
	caffeExp.set_layer_property('window_data', ['generic_window_data_param', 'source'],
			'"%s"' % prms['paths']['windowFile']['test'], phase='TEST')

	if prms['randomCrop']:
		caffeExp.set_layer_property('window_data', ['generic_window_data_param', 'random_crop'],
			'true', phase='TRAIN')
		caffeExp.set_layer_property('window_data', ['generic_window_data_param', 'source'],
			'true', phase='TEST')
	

	if prms['lossType'] == 'classify':
		for t in range(trnSz):
			caffeExp.set_layer_property('translation_fc_%d' % (t+1), ['inner_product_param', 'num_output'],
									prms['binCount'], phase='TRAIN')
		for r in range(rotSz):
			caffeExp.set_layer_property('rotation_fc_%d' % (r+1), ['inner_product_param', 'num_output'],
									prms['binCount'], phase='TRAIN')
	else:
		#Set the size of the rotation and translation layers
		caffeExp.set_layer_property('translation_fc', ['inner_product_param', 'num_output'],
								trnSz, phase='TRAIN')
		caffeExp.set_layer_property('rotation_fc', ['inner_product_param', 'num_output'],
								rotSz, phase='TRAIN')

	#Decide the slice point for the label
	#The slice point is decided by the translation labels.	
	caffeExp.set_layer_property('slice_label', ['slice_param', 'slice_point'], trnSz)	
	return caffeExp

##
def make_experiment(prms, cPrms):
	caffeExp = setup_experiment(prms, cPrms)
	caffeExp.make()
	return caffeExp	

##
def run_experiment(prms, cPrms):
	caffeExp = make_experiment(prms, cPrms)
	caffeExp.run()

	
	



	
