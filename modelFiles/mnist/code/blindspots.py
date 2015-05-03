import make_rotations as mr
import mnist_exp  as me
import numpy as np
import my_pycaffe_utils as mpu
import pdb

def get_linear_network():
	nw = []
	nw.append( [('InnerProduct', {'num_output': 10, 'nameDiff': 'ft'}), 
							('SoftmaxWithLoss', {'bottom2': 'label', 'shareBottomWithNext': True}),
							('Accuracy', {'bottom2': 'label'})] )
	return nw


def get_exp_prms(expName='exp1'):
	prms  = mr.get_prms(transform='normal', numTrainEx=60000)
	if expName == 'exp1':
		nw = get_linear_network()
		nn = nw[0]
		lastLayer = 'fc1-ft'
	else:
		raise Exception('Unrecognized experiment type')
	cPrms = me.get_caffe_prms(nn, isSiamese=False, max_iter=10000, stepsize=5000, lrAbove=None)  
	return prms, cPrms, lastLayer


def get_model_weights(prms, cPrms, modelIter, layerName):
	exp = me.make_experiment(prms, cPrms)
	return exp.get_weights(layerName, modelIter=modelIter)


def train_network(nwType='linear', lrAbove=None, max_iter=10000, stepsize=5000):
	deviceId=2
	if nwType == 'linear':
		nw = get_linear_network()

	for nn in nw:	
			prms = mr.get_prms(transform='normal', numTrainEx=60000)
			cPrms = me.get_caffe_prms(nn, isSiamese=False, lrAbove=lrAbove, max_iter=max_iter, stepsize=stepsize)
			me.run_experiment(prms, cPrms, deviceId=deviceId)


##
# Simple one layer network. 
def get_weights_experiment(expName='exp1'):
	prms, cPrms, lastLayer = get_exp_prms(expName)
	if expName == 'exp1': 
		w = get_model_weights(prms, cPrms, cPrms['max_iter'], lastLayer)
	else:
		raise Exception('Unrecognized experiment type')
	return w


def verify_null_space(expName='exp1'):
	#Get the weights and find the null space
	w     = get_weights_experiment(expName)
	w     = w.squeeze()
	U,S,V = np.linalg.svd(w)
	#Since its MNIST - the first 10 are going to the range space and the others are going
	# to the be the null space 
	nullV     = V[:,10:] 
	prms, cPrms, lastLayer = get_exp_prms(expName)
	caffeExp               = me.setup_experiment(prms, cPrms)	
	cTest = mpu.CaffeTest.from_caffe_exp_lmdb(caffeExp, prms['paths']['lmdb']['test']['im'])
	#Layers to be deleted
	delLayers = []
	delLayers = delLayers + caffeExp.get_layernames_from_type('Accuracy')
	delLayers = delLayers + caffeExp.get_layernames_from_type('SoftmaxWithLoss')
	cTest.setup_network([lastLayer], imH=28, imW=28, cropH=28, cropW=28, channels=1, 
											modelIterations=cPrms['max_iter'], delLayers=delLayers,
											batchSz=5000)

	#Input the images
	data, label,_ = cTest.get_data()
	op = cTest.net_.forward_all(blobs=[lastLayer], **{'data': data})
	op = op[lastLayer]

	#Modify the images by adding to the null space.
	perm    = np.random.permutation(nullV.shape[1])
	nullSum = 1000 * np.sum(nullV[:,perm[0:600]],axis=1)
	#nullSum  = 200 * np.random.random((784,))
	print nullSum.shape 
	#nullSum = 1000 * nullV[:,0]
	dataNew = data + nullSum.reshape((1,28,28))	
	opB = cTest.net_.forward_all(blobs=[lastLayer], **{'data': dataNew})
	opB = opB[lastLayer]
	return op, data, label, opB, dataNew


def make_mosiac(expName='exp1', isUnique=2):
	op,data,label,opB,dataNew = verify_null_space(expName)
	
	nr, nc = 5, 4
	N      = nr * nc
	im     = np.zeros((28 * nr + (nr-1)*2, 28 * nc * 2 + (nc-1)*2 + (nc-1) + 1))
	count = 0
	allConf, allConfB = [], []
	digCount  = np.zeros((10,1))
	for r in range(nr):
		for c in range(nc):
			i = r * 28 + r * 2
			j = c * 28 *2 + c * 2 + c
			while True:
				gtLabel = label[count]
				scores  = op[count].squeeze()
				scoresB = opB[count].squeeze()
				pdLabel = np.argmax(scores)
				nlLabel = np.argmax(scoresB)
				conf    = np.exp(scores - np.max(scores))
				conf    = (100.0 * conf/np.sum(conf))[pdLabel]
				confB   = np.exp(scoresB - np.max(scoresB))
				confB   = (100.0 * confB/np.sum(confB))[pdLabel]
				if pdLabel==nlLabel and conf > 99:
					if digCount[pdLabel] < 2:
						digCount[pdLabel] += 1
						break
				count += 1

			im[i:i+28, j:j+28] = data[count].squeeze()
			dNew   = dataNew[count]
			mn, mx = np.min(dNew), np.max(dNew)
			dNew   =255.0 *  (dNew.astype(float) - mn)/(mx - mn)
			print np.max(dNew), np.min(dNew)
			im[i:i+28, j+28+1:j+28+1+28] = dNew.squeeze()
			print 'GtLabel: %d, pdLabel: %d, nullLabel: %d, conf: %f, nullConf: %f'\
							% (gtLabel, pdLabel, nlLabel, conf, confB)	
			allConf.append(conf)
			allConfB.append(confB)
			count += 1
	
	allConf  = np.array(allConf)
	allConfB = np.array(allConfB) 
	print "Mean Conf: %.2f, null Conf: %.2f" % (np.mean(allConf), np.mean(allConfB))
	im = im.astype(np.uint8)
	return im


