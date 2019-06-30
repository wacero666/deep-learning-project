
#song = "TRRRAGC12903D06D5D.json"
import json
import os

def get_pairs():
    counter = 0
    y = []
    for root, dirs, files in os.walk(pth):
        for file in files:
            pth = os.path.join(root, file)
            with open(pth, 'r') as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    continue

                similar_list = data['similars']
                for p in similar_list:
                    if counter != 0:
                        print(counter)
                    if (p[0]+'.npy' in tids and file[:-5]+'.npy' in tids):
                        counter += 1
                        
                        path = "./pairs_train/%d/"%counter
                        os.makedirs(path)
                        pth1 = 'images_matrix/' + p[0]+'.npy'
                        pth2 = 'images_matrix/' + file[:-5]+'.npy'
                        print(pth1)
                        y.append((counter, float(p[1])))
                        shutil.copy(pth1, path)
                        shutil.copy(pth2, path)
    return y

if __name__== "__main__":

    pth = "lastfm/train/"
    get_pairs()
