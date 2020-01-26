from skimage import data, img_as_float
from skimage.metrics import structural_similarity as ssim
import skimage.io as sio
def ssim_score(img1,img2):
#     print(img1_dir,img2_dir)
#     img1=sio.imread(img1_dir)
    
#     img2=sio.imread(img2_dir)
    

    img1=img_as_float(img1)
    img2=img_as_float(img2)
    ssim_score=ssim(img1,img2,multi_channel=True,win_size=3)
    #print(ssim_score)
    return ssim_score
