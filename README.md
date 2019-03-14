Below you can find a outline of how to reproduce my solution for the Humpback Whale Identification competition.


#### HARDWARE REQUIREMENTS: (The following specs were used to create the original solution)
Ubuntu 16.04 LTS
128GB RAM (64GB minimum required to run keypoint matching)
Core i9-9980X CPU
8x GPU w/24GB memory each (required to run batch sizes in siamese network training)
*note that even with the above HW, training may take 13-30+ days to complete due to exhaustive brute-force keypoint matching*

#### SOFTWARE (python packages are detailed separately in `requirements.txt`):
Python 3.5.1
CUDA 10.0
cuddn 7.3

#### DATA SETUP 
```
mkdir -p whale/input
cd whale/input
```


download the competition train/test files and folders into whale/input

```
unzip mask_predictions_test_4.zip
unzip mask_predictions_train_known_4.zip
```


copy all images !=new_whale from train into the folder: train_specific
copy all images from train into folder: val

#### Keypoint matching:

```
cd kps
python index_features_2kp.py --dataset ../input/train_specific --features-db train --mask-db ../input/mask_predictions_train_known_4
python index_features_2kp.py --dataset ../input/test --features-db test --mask-db ../input/mask_predictions_test_4
python kp_matching.py
```

#### Base model training
note: for this step training is done with close attention to training/val loss and final checkpoints need to be selected at the ned of each training cycle and the
filenames put back into the training files at the appropriate insertion points
One set of train and inference files are included here.  To replicate the solution results, please substiture DenseNet121, Resnet50, and InceptionV3 as the basenetworks and run each for both train and inference (6 total runs: 3 networks x (train + inference).
```
cd ../base_network
python densenet121_200_train.py
python densenet121_422_train.py
```

#### Siamese model training
note: in this step as well, the output of the previous steps (which can't be predetermined due to checkpoint number and val_loss in file name) will need to be inserted into the py files before running
One set of train and inference files are included here.  To replicate the solution results, please substiture DenseNet121, Resnet50, and InceptionV3 as the basenetworks and run each for both train and inference (6 total runs: 3 networks x (train + inference).

```
cd ../siamese
python siamese_train.py
python siamese_inference.py
```

#### Ensemble
note: in this step as well, the output of the previous steps

```
cd ../ensemble
python ensemble.py
```
