import numpy as np
import random as r
import glob
import os

path_with_data = 'data/'
path_with_prepared_data = 'data/prep/'

def load_file(file):
    data = np.load(file)
    return data

def get_random_images(data, amount = 1000):
    array = []
    for i in range(amount):
        array.append(r.choice(data)/255)
    return array

def save_extracted_data(filename, data):
    pth = path_with_prepared_data+filename+'.npy'
    np.save(pth, data)

for file in glob.glob(os.path.join(path_with_data, '*.npy')):
    new_name = os.path.basename(file).split('.')[0]
    data = load_file(file)
    got_data = get_random_images(data, amount=500)
    save_extracted_data(new_name, got_data)


