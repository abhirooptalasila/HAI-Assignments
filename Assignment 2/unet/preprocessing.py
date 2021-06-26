import os
import numpy as np 
from PIL import Image

def normalize_image(image):
    img = Image.open(image)
    img_arr = np.asarray(img) / 255
    return img_arr

def images_mean(image_foler):
    base = os.path.abspath(image_foler)
    files = os.listdir(base)
    mean_sum = 0
    for img in files:
        img_arr = normalize_image(os.path.join(base, img))
        mean_sum += np.mean(img_arr)
    return mean_sum / len(files)

def images_std(image_foler):
    base = os.path.abspath(image_foler)
    files = os.listdir(base)
    std_sum = 0
    for img in files:
        img_arr = normalize_image(os.path.join(base, img))
        std_sum += np.std(img_arr)
    return std_sum / len(files)


if __name__=="__main__":
    par = os.path.abspath(os.getcwd())
    pat = os.path.join(par, "data", "isbi", "train-volume.tif")
    ret = normalize_image(pat)
    print(ret.shape, ret[0][:5])

    imf = os.path.join(par, "data", "isbi", "train", "images")
    print(images_mean(imf), images_std(imf))

