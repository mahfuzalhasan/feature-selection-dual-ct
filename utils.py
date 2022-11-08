import numpy as np
import pickle
import os
from sklearn.decomposition import PCA
from functools import reduce

import Config.parameters as params

def normalize(X):
    normalized_input = (X - np.amin(X)) / (np.amax(X) - np.amin(X))
    return 2*normalized_input - 1

def dump_file(file_name, image_type, data_file):
    path = os.path.join(params.saved_file_loc, image_type)
    if not os.path.exists(path):
        os.makedirs(path)

    if not os.path.exists(os.path.join(path, file_name+'.pkl')):
        with open(os.path.join(path, file_name+'.pkl'), 'wb') as f:
            pickle.dump(data_file, f)

def read_file(file_name, img_type):
    f = open(os.path.join(params.saved_file_loc, img_type, file_name+'.pkl'), 'rb')
    data = pickle.load(f)
    f.close()
    return data

def write_imp_feature_name(write_path, feat_dict):
    f = open(os.path.join(write_path), 'w')
    for key,value in feat_dict.items():
        f.write(''+ value + '\n')
    f.close()
        



def dimensionality_reduction(train_set, val_set):
    # Make an instance of the Model
    pca = PCA(n_components=0.95, random_state=2022)
    pca.fit(train_set)
    #print("explained variance: ",pca.explained_variance_ratio_)
    train_set = pca.transform(train_set)
    val_set = pca.transform(val_set)

    return train_set, val_set


def find_common_elements(features):
    res = list(reduce(lambda i, j: i & j, (set(x) for x in features)))
    #print("common features: ",res,len(res))
    return res