import cv2
import numpy as np
from keras.utils import to_categorical

def makeImageArrayFromDataBase(img_paths):
    imgs = []
    for path in img_paths: 
        img = cv2.imread(path, cv2.COLOR_BGR2RGB)
        try: 
            img = cv2.resize(img, (64,64))
            imgs.append(img)
        except: 
            print('error')
            pass #jupyter hidden file throwing an error 
    return np.array(imgs)

def convertToCategorical(y, num_classes): 
    return to_categorical(y, num_classes)

def makeImageArray(directory): 
    imgs = []
    for i in sorted(os.listdir(directory)): 
        path = os.path.join(directory, i)
        img = cv2.imread(path, cv2.COLOR_BGR2RGB)
        try: 
            img = cv2.resize(img, (128,128))
            imgs.append(img)
        except: 
            pass #jupyter hidden file throwing an error 
    return np.array(imgs)

def writeMisclassifiedSamplesToFile(filename, misclassified):
    with open(filename,"w") as f: 
        for row in misclassified: 
            f.write(str(row[0]) + ',' + str(row[1]) + '\n')