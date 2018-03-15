import numpy as np
import glob
import os
import random as r

path = 'data/prep/'
prep = []
learn_data_array = []
learn_target_array = []

path, sth, files = next(os.walk(path))
num_of_files = len(files)
answer_array = []

def load(file):
    data = np.load(file).astype(np.float32)
    return data

def prep_answer_array(element):
    target = [0]*num_of_files
    target[len(answer_array)] = 1
    answer_array.append([element, np.argmax(target)])
    return target

for filename in glob.glob(os.path.join(path, '*.npy')):
    base_name = os.path.basename(filename).split('.')[0]
    target = prep_answer_array(base_name)
    array = load(filename)
    for item in array:
        prep.append([item, target])

r.shuffle(prep)
for item in prep:
    learn_data_array.append(item[0])
    learn_target_array.append(item[1])

indicator = 0
splited_learn_data = np.split(np.asarray(learn_data_array), 10)
splited_target_data = np.split(np.asarray(learn_target_array), 10)
print('splited data: ',len(splited_learn_data),len(splited_target_data))

def get_batch():
    global indicator
    indicator += 1
    if indicator >= len(splited_target_data):
        indicator = 0

    return splited_learn_data[indicator], splited_target_data[indicator]

def get_whole_data():
    return learn_data_array, learn_target_array

def get_data_to_test(a):
    return learn_data_array[-a:]
def get_answers(a):
    return  learn_target_array[-a:]