import numpy as np
import pickle
import os
from sklearn.decomposition import PCA
from functools import reduce
import pandas as pd
from statistics import mean

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

import Config.parameters as params
import Config.configuration as cfg

def normalize(X):
    normalized_input = (X - np.amin(X)) / (np.amax(X) - np.amin(X))
    return 2*normalized_input - 1

def random_shuffle(data, label):
    indices = np.random.permutation(data.shape[0])
    data = data[indices, :, :]
    label = label[indices]
    return data, label

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


def plot_matplotlib(x_coords, y_coords, title, x_label, y_label, feature_name, save_path):
    plt.figure()
    plt.scatter(x_coords, y_coords)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label) 
    plt.savefig(os.path.join(save_path, feature_name+'.png'))
    plt.close()

def feature_stats(features):
    #for feature in features:
    save_path = os.path.join(cfg.output_dir, "feature_stats")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    #exit()
    
    for _, (feature_name, val) in enumerate(features):
        print("feature: ",feature_name)
        energy_feature_dict = {i:[] for i in range(40, 141, 5)}
        mapper = {i:list(energy_feature_dict.keys())[i] for i in range(21)}
        #print("mapper: ",mapper)
        #print("energy_f_dict: ",energy_feature_dict)
        xs = []
        ys = []
        for patient_file in  os.listdir(cfg.data_folder):
            patient_id = patient_file[:patient_file.rindex('.')]
            #print("patient id: ",patient_id)
            df = pd.read_csv(os.path.join(cfg.data_folder, patient_file))
            feature_vals = df.loc[:, feature_name]
            for j, vals in enumerate(feature_vals):
                xs.append(mapper[j])
                ys.append(vals)
                energy_feature_dict[mapper[j]].append(vals)
        xs_unique = []
        ys_avg = []
        for energy_level, vals in energy_feature_dict.items():
            xs_unique.append(energy_level)
            ys_avg.append(mean(vals))


        plot_matplotlib(xs, ys, feature_name, "energy level", "feature for each patient", feature_name, save_path)
        plot_matplotlib(xs_unique, ys_avg, "average "+feature_name, "energy level", "avg feature values", feature_name+'_avg', save_path)
        
    #lists = sorted(energy_feature_dict.items()) # sorted by key, return a list of tuples
    #x, y = zip(*lists) # unpack a list of pairs into two tuples
    #print(x, y)
    
        



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