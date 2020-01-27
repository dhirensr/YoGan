import numpy as np
from random import shuffle
import os
import sys
import matplotlib.pyplot as plt
from PIL import Image
import datetime,pickle
from skimage.io import imsave
from yoga_text_to_class_predict import load_text_model
from keras_text_to_image.library.utility.ssim import ssim_score
from keras.preprocessing.image import img_to_array, load_img

YOGA_CLASS=['agnistambhasana', 'ananda balasana', 'ardha pincha mayurasana',
       'ardha uttanasana', 'astavakrasana', 'bhekasana', 'bhujapidasana',
       'bitilasana', 'camatkarasana', 'chaturanga dandasana',
       'ganda bherundasana', 'garudasana', 'halasana', 'malasana',
       'marichyasana iii', 'marjaryasana', 'matsyasana',
       'parivrtta janu sirsasana', 'parivrtta trikonasana', 'parsva bakasana',
       'parsvottanasana', 'pasasana', 'salamba bhujangasana',
       'salamba sarvangasana', 'savasana', 'sukhasana',
       'supta baddha konasana', 'tolasana', 'urdhva mukha svanasana',
       'ustrasana', 'uttana shishosana', 'utthita parsvakonasana', 'vajrasana',
       'virabhadrasana iii']

gan_list=['14012020_2131_45','epoch-4950-03012020_1216_55','15012020_2205_37','16012020_1756_32','16012020_2257_39'] #replace with final 5 class models

img_width = 64
img_height = 64
model_dir_path = '/dev/shm/shashank3110'+ '/final_models'
ground_truth_img_dir = 'keras-text-to-image/demo/data/yoga/img/'
# txt_dir_path = 'keras-text-to-image/demo/data/yoga/test_txt/'
for model_fname in os.listdir(model_dir_path):
    gan_list.append(model_fname.split('h5')[0])

texts = dict()
names = []



class_list=['pasasana','agnistambhasana','bhujapidasana','bitilasana','matsyasana']
model_dict = dict(zip(class_list,gan_list))
dt_string = datetime.datetime.now().strftime("%d%m%Y_%H%M_%S")
pickle_path="/home/shashank3110/keras-text-to-image/demo/models/cv_pickle.pk"
with open(pickle_path,"rb") as f:
    cv=pickle.load(f)


gan = DCGan()

texts = []
for text in texts:
    ssim_scores=[]
    class_predicted = load_text_model(text,pickle_path,YOGA_CLASS)
    print(f"Model chosen is {class_predicted,model_dict[class_predicted]}")
    gan.load_model(model_dir_path,model_dict[class_predicted])
    img_path = ground_truth_img_dir  +  class_predicted + '.png'

    true_img = img_to_array(load_img(img_path, target_size=(img_width, img_height)))
    true_img = (true_img.astype(np.float32) / 255) * 2 - 1
    for model_name in gan_list:
        gan.load_model(model_dir_path,model_name)
        generated_image = gan.generate_image_from_text(text)
        ssim_scores.append([generated_image,ssim_score(true_img,generated_image)])
    predicted_image,score=max(ssim_scores, key = lambda x: x[1])
    print(score)
