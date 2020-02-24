'''
Data augmentation  script using Keras ImageDataGenerator.

'''
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
from matplotlib import pyplot as plt
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from matplotlib import pyplot
from numpy import expand_dims

def augment_images(image_directory,fpath,fname,batch_size):
	
    '''
    Augmentations used: Rotation, horizontal flip,zoom,vertical shift.
    
    '''
    im_gen = ImageDataGenerator(rotation_range=10,
                                height_shift_range=.15,
                                horizontal_flip=True,
                                zoom_range=[0.75,1.0]) 
    if len(fname)==0:
        return
    img = load_img(fpath+fname+'.png')
    data = img_to_array(img)
    samples = expand_dims(data, 0)
    it=im_gen.flow(samples,batch_size=batch_size)
    for i in range(4):
	    batch = it.next()
	    # convert to unsigned integers for viewing
	    image = batch[0].astype('uint8')
	    # plot raw pixel data
	    pyplot.imsave(image_directory+fname+str(i)+'.png',image)


def generate():
    '''
    Iterates through each yoga pose directory and applies augmentation.
    '''
    path = '/yoga/img_aug_test/'
    pwd = os.getcwd()
    read_directory=pwd+'/yoga/img/'
    yoga_dir =os.listdir(pwd+path)
    for yoga_pose in os.listdir(read_directory):
        fname= yoga_pose.split(".")[0]
        augment_images(pwd+path+fname+"/",read_directory,fname,1)




if __name__ == '__main__':
    generate()
