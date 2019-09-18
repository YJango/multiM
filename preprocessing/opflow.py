from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# from __future__ import unicode_literals
import numpy as np
from PIL import Image
import time
import argparse
import pyflow
import os,sys
import subprocess
import cv2
import os.path

def flow_twoimages(im1path,im2path,outpath,outimage=False):
    im1 = np.array(Image.open(im1path))
    im2 = np.array(Image.open(im2path))
    im1 = im1.astype(float) / 255.
    im2 = im2.astype(float) / 255.

    # Flow Options:
    alpha = 0.012
    ratio = 0.75
    minWidth = 20
    nOuterFPIterations = 7
    nInnerFPIterations = 1
    nSORIterations = 30
    colType = 0  # 0 or default:RGB, 1:GRAY (but pass gray image with shape (h,w,1))

    s = time.time()
    u, v, im2W = pyflow.coarse2fine_flow(
        im1, im2, alpha, ratio, minWidth, nOuterFPIterations, nInnerFPIterations,
        nSORIterations, colType)
    e = time.time()
    print('Time Taken: %.2f seconds for image of size (%d, %d, %d)' % (
        e - s, im1.shape[0], im1.shape[1], im1.shape[2]))
    flow = np.concatenate((u[..., None], v[..., None]), axis=2)
    np.save(outpath, flow)
    if outimage:
        hsv = np.zeros(im1.shape, dtype=np.uint8)
        hsv[:, :, 0] = 255
        hsv[:, :, 1] = 255
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        cv2.imwrite(outpath+'.png', rgb)
        cv2.imwrite(outpath+'.jpg', im2W[:, :, ::-1] * 255)
        
def get_filenames(path,shuffle=False,extension='.mp4'):
    # get all file names 
    files= os.listdir(path) 
    filepaths = [path+file for file in files if not os.path.isdir(file) and extension in file]
    # shuffle
    if shuffle:
        ri = np.random.permutation(len(filepaths))
        filepaths = np.array(filepaths)[ri]
    #print(filepaths)
    return filepaths
PATH='/USERS/d8182103/firstimpressionV2/image_320x180/'
newPATH = '/home/hi-lab/Documents/'
for s in ['train/','test/','vali/']:
    for f in get_filenames(PATH+s,shuffle=False,extension=''):
        for p in get_filenames(f+'/',shuffle=False,extension=''):
            images = ['%s/320x180-%s.jpeg' %(p,i) for i in range(1,461,2)]
            for e,m in enumerate(images):
                if e!=230 and os.path.isfile(m):
                    newp = p.split('/')[-1]
                    newf = f.split('/')[-1]
                    outpath = '%s%s-%s-%s-%s' %(newPATH,s,newf,newp,e+1)
                    print(outpath)
                    my_file = Path(outpath+".png")
                    if os.path.isfile(m)!=True:
                        flow_twoimages(images[e],images[e+1],outpath)