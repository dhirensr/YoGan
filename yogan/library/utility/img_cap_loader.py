'''
Utility file to load normalizd images and the text descriptions.
'''

import os
import numpy as np
from keras.preprocessing.image import img_to_array, load_img
import matplotlib.pyplot as plt
import os
import cv2
import random
from PIL import Image

pwd=os.getcwd()
def load_normalized_img_and_its_text(img_dir_path, txt_dir_path, img_width, img_height):
    '''
    Iterates through image and text directories and pairs them up.
    '''
    fnames= os.listdir(img_dir_path)
    images = dict((f,[]) for f in fnames)
    texts = dict()

    for yoga in fnames:
        for f in os.listdir(img_dir_path+'/'+yoga):
            filepath = os.path.join(img_dir_path+'/'+yoga, f)

            if os.path.isfile(filepath) and f.endswith('.png'):
                name = f.replace('.png', '')
                images[yoga].append(filepath)

    for f in os.listdir(txt_dir_path):
        filepath = os.path.join(txt_dir_path, f)
        if os.path.isfile(filepath) and f.endswith('.txt'):
            name = f.replace('.txt', '')
            texts[name] = open(filepath, 'rt').read()

    result = []

    for name, img_path_list in images.items():
        # If training single class models then pass only a single class name below.
        if(name in ['pasasana','bhujapidasana','bitilasana','matsyasana','agnistambhasana']): #load only these 5 classes as we have trained for 5 classes for now
            for img_path in img_path_list:
            #print(f"img_path={img_path}, name={name}")
                if name in texts:
                    text = texts[name]
                    #image = Image.open(img_path).convert('RGB')
                    #print(type(image))
                    image = img_to_array(load_img(img_path, target_size=(img_width, img_height)))
                    image = (image.astype(np.float32) / 255) * 2 - 1

                    result.append([image, text])

    random.shuffle(result)
    return np.array(result)
