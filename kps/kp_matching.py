from imutils.feature import FeatureDetector_create, DescriptorExtractor_create
import glob
import numpy as np
import cv2
from imutils.feature import DescriptorMatcher_create
from tqdm import tqdm
import time
import numpy as np
import faiss
from numba import njit
import pandas as pd
import shutil
import pickle
import sklearn


#fast implementation of matching using njit
@njit
def goodmatchmaker(rawMatches, kpmask):
    goodMatches = 0
    for m in range(len(rawMatches)):
        if rawMatches[m][0] < 0.64*rawMatches[m][1]:
            #matchesMask[i]=[1,0]
            goodMatches +=1
            kpmask[m] = 1
    return goodMatches, kpmask

#files needed which include kp index data
query_kps = np.load('test_kpsI.npy')
query_features = np.load('test_featsI.npy')
query_indexes = np.load('test_idxsI.npy')
query_images = np.load('test_imgsI.npy')

index_kps = np.load('train_kpsI.npy')
index_features = np.load('train_featsI.npy')
index_indexes = np.load('train_idxsI.npy')
index_images = np.load('train_imgsI.npy')


#gbt model to validate homography matrix
with open('final_model_gb.pkl', 'rb') as f:
    model = pickle.load(f)

#matcher = DescriptorMatcher_create("BruteForce")
t0total=time.time()

ratio=0.7
results = {}
timearray = []
k=2
d=128

matches = []
randsacmatches = []
ran_ratios = []
Mdummy = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])  #initialize dummy homography array
train_match = []
test_match = []
range_array = []
uniques = []


for j in tqdm(range(1400)):
 
    (startq, endq) = query_indexes[j]
    queryKps = query_kps[startq:endq]
    queryDescs = query_features[startq:endq]
    
    res = faiss.StandardGpuResources()  # use a single GPU

    # build a flat (CPU) index
    index_flat = faiss.IndexFlatL2(d)
    # make it into a gpu index
    #gpu_index_flat = faiss.index_cpu_to_gpu(res, 0, index_flat)  #for single gpu
    gpu_index_flat = faiss.index_cpu_to_all_gpus(index_flat)   #for multigpu
    
    gpu_index_flat.add(queryDescs) 

    #start the search, loop over index
    for coverPath in range(len(index_indexes)):
        (start, end) = index_indexes[coverPath]
        kps = index_kps[start:end]
        descs = index_features[start:end]

        t0 = time.time()
        rawMatches, I = gpu_index_flat.search(descs, k)  #faiss gpu search
        timearray.append(time.time()-t0)

        kpmask = np.zeros((len(rawMatches),), dtype=int)
        goodMatches, kpmask = goodmatchmaker(rawMatches, kpmask)
        matches.append(goodMatches)
        randsacmatch = 0
        ran_ratio=0.01
        M = np.copy(Mdummy)
        if goodMatches >9:
            src_pts = kps[kpmask==1].reshape(-1,1,2)  #gets correct kps
            dst_pts = queryKps[I[kpmask==1][:,0]].reshape(-1,1,2)  #gets correct querykps
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RHO,5.0, maxIters = 20000)
            randsacmatch = np.sum(mask)
            
            if randsacmatch > 1:
                src_pts = src_pts[mask==1]
                dst_pts = dst_pts[mask==1]
                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,9.0, maxIters = 20000)
                randsacmatch = np.sum(mask)
            
        if M is not None:
            M = M.flatten()
            M = M[:8]
            M[~np.isfinite(M)] = 0.
            pred = model.predict(M.reshape(1,-1))
            homog_dispo = pred[0]
            if homog_dispo==1:
                randsacmatches.append(randsacmatch)  
                train_match.append(coverPath)
                test_match.append(j)
                uniques.append(len(np.unique(dst_pts, axis=0)))
        
    
print('Total exectution time: {}'.format(time.time()- t0total))    

df = pd.DataFrame({'matches': randsacmatches, 'train_match': train_match, 'test_match': test_match, 'uniques': uniques})

df.to_csv('kp_matches.csv')
