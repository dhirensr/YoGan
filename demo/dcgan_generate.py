import numpy as np
from random import shuffle
import os
import sys
import matplotlib.pyplot as plt
from PIL import Image
import datetime,pickle
from skimage.io import imsave
from yoga_text_to_class_predict import load_text_model
pickle_path="/home/shashank3110/keras-text-to-image/demo/models/cv_pickle.pk"
with open(pickle_path,"rb") as f:
    cv=pickle.load(f)
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
pwd=os.getcwd()
dt_string = datetime.datetime.now().strftime("%d%m%Y_%H%M_%S")
def main():
    current_dir = os.path.dirname(__file__)
    sys.path.append(os.path.join(current_dir, '..'))
    current_dir = current_dir if current_dir is not '' else '.'

    img_dir_path = current_dir + '/data/yoga/img_aug_test/' #/data/yoga/img_aug'
    txt_dir_path = current_dir + '/data/yoga/txt'
    model_dir_path = '/dev/shm/shashank3110'+ '/final_models'
    #model_dir_path = current_dir + '/models'

    img_width = 64
    img_height = 64

    from keras_text_to_image.library.dcgan import DCGan
    from keras_text_to_image.library.utility.image_utils import img_from_normalized_img
    from keras_text_to_image.library.utility.img_cap_loader import load_normalized_img_and_its_text
    from keras_text_to_image.library.utility.ssim import ssim_score

    image_label_pairs = load_normalized_img_and_its_text(img_dir_path, txt_dir_path, img_width=img_width, img_height=img_height)
    #print(f"image label pairs={image_label_pairs}")
    #print(f"Result length of image label pairs={len(result)}")
    #for i in range(10):
    shuffle(image_label_pairs)

    #gan = DCGan()

    gan = DCGan()
    #gan2 = DCGan()
    #gan3 = DCGan()
    #gan4 = DCGan()
    #gan5 = DCGan()

    #gan1.load_model(model_dir_path,'14012020_2131_45') #bitilasana
    #gan2.load_model(model_dir_path,'epoch-4950-03012020_1216_55') # pasasana
    #gan3.load_model(model_dir_path,'15012020_2205_37') # agnistambhasana
    #gan4.load_model(model_dir_path,'16012020_1756_32') #bhujapidasana
    #gan5.load_model(model_dir_path,'16012020_2257_39') # matsyasana
    #gan.load_model(model_dir_path)
    #gan_list=[gan1,gan2,gan3,gan4,gan5]
    gan_list=['14012020_2131_45','epoch-4950-03012020_1216_55','15012020_2205_37','16012020_1756_32','16012020_2257_39']

    ### Added on 19/01/2020
    class_list=['pasasana','agnistambhasana','bhujapidasana','bitilasana','matsyasana']
    model_dict = dict(zip(class_list,gan_list))
    ###
    print(len(image_label_pairs))

    for i in range(len(image_label_pairs)):#np.random.randint(low=0,high=339,size=10):

        image_label_pair = image_label_pairs[i]
        normalized_image = image_label_pair[0]
        text = image_label_pair[1]


        #print(text)
        image = img_from_normalized_img(normalized_image)
        normalized_image_dir=os.getcwd() + '/data/outputs/' + 'normalized/' + 'generated-' + str(i) +"-"+dt_string + '-0.png'
        #print(current_dir + '/data/outputs/')

        image.save(normalized_image_dir)
        ### Added on 19/01/2020
        class_predicted = load_text_model(text,pickle_path,YOGA_CLASS)
        print(f"Model chosen is {class_predicted,model_dict[class_predicted]}")
        gan.load_model(model_dir_path,model_dict[class_predicted])

        generated_images = gan.generate_image_from_text(text)
        generated_images.save(current_dir + '/data/outputs/models_output/' + DCGan.model_name + '-generated-image-NN-Model-' + str(i) + '-'  +"-"+dt_string+ '.png')
        ###
        k=0
        generated_images_dir=[]
        ssim_scores=[]
        #for j in range(1):
        for model_name in gan_list:
            gan.load_model(model_dir_path,model_name)
            generated_images = gan.generate_image_from_text(text)
            generated_image_dir=os.getcwd() + '/data/outputs/models_output/' + DCGan.model_name + '-generated-test-single_class-test' + str(i) +str(k)+ '-'  +"-"+dt_string+ '.png'
            generated_images.save(generated_image_dir)
            generated_images_dir.append(generated_image_dir)
            k+=1
        #print(normalized_image_dir,generated_images_dir)

        for j in generated_images_dir:
            ssim_scores.append([j,ssim_score(normalized_image_dir,j)])
        predicted_image,score=max(ssim_scores, key = lambda x: x[1])
        print(predicted_image,score)
        print(f"Model chosen is {class_predicted,model_dict[class_predicted]}")


if __name__ == '__main__':
    main()
