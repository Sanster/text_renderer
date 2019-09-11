import sys
sys.path.append(r'C:\jianweidata\ocr')


"""
缩小 再扩大
"""
import glob
import cv2
import os 
import numpy as np 

interpolation = {'LINEAR':cv2.INTER_LINEAR , 'NEAREST':cv2.INTER_NEAREST , 'AREA':cv2.INTER_AREA }

def pydown(img,interpolation,scale):
    img = cv2.resize(img,None,fx = scale ,fy = scale, interpolation = interpolation)
    img = cv2.resize(img,(256,32),interpolation = interpolation)
    return img 

def apply_gauss_blur(img, ks=[3,5]):
    if ks is None:
        ks = [7, 9, 11, 13]
    ksize = np.random.choice(ks)

    sigmas = [0, 1, 2, 3, 4, 5, 6, 7]
    sigma = 0
    if ksize <= 3:
        sigma = np.random.choice(sigmas)
    img = cv2.GaussianBlur(img, (ksize, ksize), sigma)
    return img


srcpath = r'C:\jianweidata\ocr\test\src'
dstpath = r'C:\jianweidata\ocr\test\dst'

imgfiles = glob.glob(os.path.join(srcpath,'*.jpg'))
#for file in imgfiles:
#    basename = os.path.basename(file).split('.')[0]
#    img = cv2.imread(file)
#    for scale in [0.8,0.9,1.1,1.25,1.35]:
#        cimg = np.copy(img)
#        cimg = pydown(cimg,cv2.INTER_NEAREST,scale)
#        cv2.imwrite(os.path.join(dstpath,basename)+'_'+'NEAREST'+str(scale)+'.jpg',cimg)


#for file in imgfiles:
#    basename = os.path.basename(file).split('.')[0]
#    img = cv2.imread(file)
#    for sigma in np.arange(0.1,1.01,0.1):
#        cimg = np.copy(img)
#        cimg = cv2.GaussianBlur(cimg, (0,0),sigma)
#        cv2.imwrite(os.path.join(dstpath,basename)+'_'+str(int(sigma*10))+'.jpg',cimg)


for file in imgfiles:
    basename = os.path.basename(file).split('.')[0]
    img = cv2.imread(file)
    for sigma in np.arange(1,5):
        cimg = np.copy(img)
        cimg = cv2.blur(cimg, (sigma,sigma))
        cv2.imwrite(os.path.join(dstpath,basename)+'_'+str(int(sigma))+'.jpg',cimg)