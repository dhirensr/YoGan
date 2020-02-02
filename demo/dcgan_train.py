import sys
import os
import numpy as np
from random import shuffle


def main():

    current_dir = os.path.dirname(__file__)
    sys.path.append(os.path.join(current_dir, '..'))
    current_dir = current_dir if current_dir is not '' else '.'

    img_dir_path = current_dir + '/data/yoga/img_aug'
    txt_dir_path = current_dir + '/data/yoga/txt'
    model_dir_path = current_dir + '/models'

    img_width = 64#256
    img_height = 64#256 large size of 256 leads to  ram alloc error
    img_channels = 3

    from yogan.library.dcgan import DCGan
    from yogan.library.utility.img_cap_loader import load_normalized_img_and_its_text

    image_label_pairs = load_normalized_img_and_its_text(img_dir_path, txt_dir_path, img_width=img_width, img_height=img_height)
    shuffle(image_label_pairs)

    gan = DCGan()
    gan.img_width = img_width
    gan.img_height = img_height
    gan.img_channels = img_channels
    gan.random_input_dim = 100
    gan.glove_source_dir_path = './very_large_data'

    batch_size = 51
    epochs = 8000
    print("epochs,batch_size",epochs,batch_size)
    gan.fit(model_dir_path=model_dir_path, image_label_pairs=image_label_pairs,
            snapshot_dir_path=current_dir + '/data/snapshots',
            snapshot_interval=100,
            batch_size=batch_size,
            epochs=epochs)


if __name__ == '__main__':
    main()
