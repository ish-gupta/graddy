import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def plot_cam(image_path, superimposed_img):

    img = readImage(image_path)
    img = np.expand_dims(img,axis=0)

    superimposed_img = grad_cam_heatmap(path, layer_index)

    f, axarr = plt.subplots(1,2,figsize=(20, 30))
    axarr[0].imshow(img[0])
    axarr[0].set_title(name)
    axarr[1].imshow(superimposed_img)
    axarr[1].set_title(name)
    plt.show()