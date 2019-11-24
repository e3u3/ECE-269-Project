# As usual, some setup code
import numpy as np
import os
import matplotlib.pyplot as plt

def load_image(ROOT):
    """ Load the image dataset from disk """
    xn = []
    xs = []
    for b in range(1,3):
        path = os.path.join(ROOT, 'Part_%d' % (b, ))
        X, Y, original_size = load_part(path,b)
        xn.append(X)
        xs.append(Y)
    del X, Y
    XN = np.concatenate(xn)
    XS = np.concatenate(xs)
    return XN, XS, original_size

def load_part(filepath,num):
    """ load neutral and smiling image seperately """
    X = []
    Y = []
    original_size = []
    for i in range((num-1)*100+1,num*100+1):
        # Read image of neutral
        img_1 = plt.imread(filepath + '/%da.jpg' % (i, ))
        original_size = np.shape(img_1)
        X.append(np.array(img_1).flatten())
        
        # Read image of smiling
        img_2 = plt.imread(filepath + '/%db.jpg' % (i, ))
        Y.append(np.array(img_2).flatten())
    return X, Y, original_size