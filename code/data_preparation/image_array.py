from scipy import misc, ndimage
import os
import pickle
import numpy as np
def store_spectogram():
    images = []
    track_ids = []
    i = 0
    for root, dirnames, filenames in os.walk("./images/"):
        for filename in filenames:
            filepath = os.path.join(root, filename)
            
            track_ids.append(filename[:-4])
            image = ndimage.imread(filepath, mode='RGB') 
            image_resized = misc.imresize(image, (64, 120))
            print("file", i, ": track id:", filepath)
            i += 1
            np.save('./images_matrix/%s' %filename[:-4], image_resized)

if __name__== "__main__":
    store_spectogram()