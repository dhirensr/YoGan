'''
Utility script to generate  Structural Similarity between 
generated and training images to evaluate 
image quality.
'''

from skimage import data, img_as_float
from skimage.metrics import structural_similarity as ssim
import skimage.io as sio
def ssim_score(img1,img2):
    img1=img_as_float(img1)
    img2=img_as_float(img2)
    ssim_score=ssim(img1,img2,multi_channel=True,win_size=3)
    return ssim_score
