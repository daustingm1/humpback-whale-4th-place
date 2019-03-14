# USAGE
# python index_features_2kp.py --dataset ../input/train_specific --features-db train --mask-db ../input/mask_predictions_train_known_4
# python index_features_2kp.py --dataset ../input/test --features-db test --mask-db ../input/mask_predictions_test_4


from tqdm import tqdm
from imutils.feature import FeatureDetector_create, DescriptorExtractor_create
from imutils import paths
import argparse
import imutils
import cv2
import os
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="Path to the directory that contains the images to be indexed")
ap.add_argument("-f", "--features-db", required=True, help="either test or train")
ap.add_argument("-m", "--mask-db", required=True, help="Path to where the masks will be stored")
args = vars(ap.parse_args())

def describe_kps(image, mask): 
    kps = detector.detect(image, mask=mask)
    (kps, descs) = descriptor.compute(image, kps)
    if len(kps) == 0:
        return (None, None)
    kps = np.float32([kp.pt for kp in kps])
    return (kps, descs)

# the detector and descriptor used for analysis
detector = FeatureDetector_create("SIFT")
descriptor = DescriptorExtractor_create("RootSIFT")

start=0
indexlist = []  #list of indexes that relate to the rows of the kps for a given image
imagelist = []  #list of image files
kplist = [] #vector of length 130.  1st two numbers are the coordinates, last 128 are the rootsift descriptor for the given point
for (i, imagePath) in tqdm(enumerate(paths.list_images(args["dataset"]))):
    #read the image and mask
    filename = imagePath[imagePath.rfind("/") + 1:]
    image = cv2.imread(imagePath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = cv2.imread(args["mask_db"] + '/' + os.path.basename(imagePath)[:-4]+'.png', 0)
    mask[mask>0]=1
    
    #if the mask is small, just use the entire image.  helps reduce false positives due to poor mask
    if np.sum(mask) < 1000:
        mask[mask==0] = 1
    # describe the image
    (kps, descs) = describe_kps(image, mask)
    # if either the keypoints or descriptors are None, then ignore the image
    if kps is None or descs is None:
        continue
    end = start+len(kps)
    indexlist.append((start, end))
    imagelist.append(filename)
    kplist.extend(np.hstack([kps, descs]))
    start = end

kps_all = np.array(kplist)
coordinate_array = kps_all[:,:2]
features_array = kps_all[:,2:]

#write the arrays to numpy files and save
np.save('{}_imgsI.npy'.format(args["features_db"]), np.array(imagelist))
np.save('{}_idxsI.npy'.format(args["features_db"]), np.array(indexlist))
np.save('{}_featsI.npy'.format(args["features_db"]), features_array)
np.save('{}_idxsI.npy'.format(args["features_db"]), coordinate_array)
