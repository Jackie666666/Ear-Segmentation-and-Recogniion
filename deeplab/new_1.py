from PIL import Image
import numpy as np
import cv2
import glob
import os

os.chdir(r'C:\Users\kshit\Desktop\SegmentationClass')

def convert_img(img):
    shape = img.shape
    result = np.zeros(shape)
    for x in range(0,shape[0]):
        for y in range(0,shape[1]):
            if img[x,y]==False:
                result[x,y]=0
            elif img[x,y]==True:
                result[x,y]=1
    return result
	


count = 0
for filenames in glob.glob('**/*.png',recursive=True):
    count = count+1
    if count == 100:
        print(count)
        count=0
    im = np.array(Image.open(filenames))    
    result = convert_img(im)
    cv2.imwrite(r'C:\Users\kshit\Desktop\SegmentationClass_1\\'+filenames,result)	