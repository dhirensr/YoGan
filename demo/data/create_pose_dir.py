'''
One time run script to create test/train directories before image augmentation
has been run to generate train and test images.
1. Run this script.
2. Run augment_data.py to generate augmented training samples.
3. Run augment_data.py to generate augmented test samples.
'''
import os 
path='yoga/img/'
aug_path='yoga/'
train='img_aug_train/'
test='img_aug_test/'
dir= os.listdir(path)

for pose_name in dir:
    p=pose_name.split(".")[0]
    if len(p)>0:
        os.mkdir(aug_path+train+p)
        os.mkdir(aug_path+test+p)
