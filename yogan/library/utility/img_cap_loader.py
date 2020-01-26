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
#     i=0
    #fnames=['pasasana']#,'bitilasana','agnistambhasana','bhujapidasana','matsyasana'],'bitilasana','ananda balasana','astavakrasana','ardha uttanasana'
    for name, img_path_list in images.items():
        #print(f"img_path_list={img_path_list}")
#         if i==1:
#             break
        if(name in ['matsyasana']): #'pasasana','bhujapidasana','bitilasana','matsyasana''agnistambhasana'
            for img_path in img_path_list:
            #print(f"img_path={img_path}, name={name}")
                if name in texts:
                    text = texts[name]
                    #image = Image.open(img_path).convert('RGB')
                    #print(type(image))
                    image = img_to_array(load_img(img_path, target_size=(img_width, img_height)))
                    image = (image.astype(np.float32) / 255) * 2 - 1
                    #image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                    result.append([image, text])
#         i+=1


    #print(len(result))
    random.shuffle(result)
    return np.array(result)

# def load_normalized_img_and_its_text(img_dir_path, txt_dir_path, img_width, img_height):

#     images = dict()
#     texts = dict()

#     for f in os.listdir(img_dir_path):
#         filepath = os.path.join(img_dir_path, f)

#         if os.path.isfile(filepath) and f.endswith('.png'):
#             name = f.replace('.png', '')
#             images[name] = filepath

#     for f in os.listdir(txt_dir_path):
#         filepath = os.path.join(txt_dir_path, f)
#         if os.path.isfile(filepath) and f.endswith('.txt'):
#             name = f.replace('.txt', '')
#             texts[name] = open(filepath, 'rt').read()

#     result = []
# #     i=0
#     for name, img_path in images.items():
# #         if i==1:
# #             break
#         if name in texts:
#             text = texts[name]
#             image = img_to_array(load_img(img_path, target_size=(img_width, img_height)))
#             image = (image.astype(np.float32) / 255) * 2 - 1
#             result.append([image, text])
# #             i+=1

#     return np.array(result)
