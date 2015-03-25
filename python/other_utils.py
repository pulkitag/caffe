## @package other_utils
#  Miscellaneous Util Functions
#

##
# Verify if all the keys are present recursively in the dict
def verify_recursive_key(data, keyNames):
	'''
		data    : dict like data['a']['b']['c']...['l']
		keyNames: list of keys 
	'''
	assert isinstance(keyNames, list), 'keyNames is required to be a list'
	assert data.has_key(keyNames[0]), '%s not present' % keyNames[0]
	for i in range(1,len(keyNames)):
		dat = reduce(lambda dat, key: dat[key], keyNames[0:i], data)
		assert isinstance(dat, dict), 'Wrong Keys'
		assert dat.has_key(keyNames[i]), '%s key not present' % keyNames[i]
	return True

##
# Set the value of a recursive key. 
def set_recursive_key(data, keyNames, val):
	if verify_recursive_key(data, keyNames):
		dat = reduce(lambda dat, key: dat[key], keyNames[:-1], data)
		dat[keyNames[-1]] = val
	else:
		raise Exception('Keys not present')
