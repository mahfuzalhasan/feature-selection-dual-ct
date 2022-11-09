from data_process import DataProcess
import Config.configuration as cfg
from utils import *

import numpy as np
from sklearn.svm import LinearSVC, SVC
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA

from sklearn.utils.class_weight import compute_class_weight 
from sklearn.linear_model import LogisticRegression

import os

    


def important_feature_selection(train_features, val_features, initial_feature_names):
    # 10 samples with 5 features
    #train_features = np.random.rand(10,5)

    pca = PCA(n_components=0.95, svd_solver="full", random_state=1)
    pca.fit(train_features)
    X_pc = pca.transform(train_features)
    X_val = pca.transform(val_features)
    # 90 patients --> 90x4019

    # 90 patients --> 90x3x4019 
    # to run PCA --> 90 x 12057 ---> [0 5000]

    #print("after pca feature shape: ",X_pc.shape, X_val.shape)

    # number of components
    print("pca_components: ", pca.components_.shape)
    
    n_pcs = pca.components_.shape[0]
    #print("explained variance: ", pca.explained_variance_, pca.explained_variance_ratio_)
    #print("number of components: ", n_pcs)

    # get the index of the most important feature on EACH component
    # LIST COMPREHENSION HERE
    most_important = [np.abs(pca.components_[i]).argmax() for i in range(n_pcs)]
    print("first most imp comp: ",most_important[0])
    first_comp = pca.components_[:, most_important[0]]

    #print("pca components: ", first_comp)

    #print("X_pc values: ", X_pc[:, 0])
    
    
    
    #print("train fratures ", train_features[:, most_important[0]])
    #print(X_pc[])


    # list to capture all feature names
    #initial_feature_names = ['a','b','c','d','e']
    
    # 80x4019 --> PCA --> min(num_sample, num_feature) --> 80
    # get the names selected by PCA
    most_important_names = [initial_feature_names[most_important[i]%4019] for i in range(n_pcs)]

    # LIST COMPREHENSION HERE AGAIN
    dic = {'PC{}'.format(i): most_important_names[i] for i in range(n_pcs)}
    print(dic)
    #exit()
    return X_pc, X_val, dic

def random_shuffle(data, label):
    indices = np.random.permutation(data.shape[0])
    data = data[indices, :, :]
    label = label[indices]
    return data, label

def select_keV_data(data, input_keV, mapper):
    selected_data = []
    for keV in input_keV:
        d = data[:, mapper[keV], :]
        selected_data.append(d)
    selected_data = np.stack(selected_data, axis=1)
    return selected_data




if __name__=='__main__':

    img_kev_list = [i for i in range(40, 141, 5)]
    """ feat_appearances = {}

    for energy_level in img_kev_list:
        f = open(os.path.join(cfg.output_dir, str(energy_level)+'.txt'), "r")
        data = f.read()
        list_data = list(set(data.split("\n")[:-1]))

        for feat in list_data:
            if feat not in feat_appearances:
                feat_appearances[feat] = 0
            else:
                feat_appearances[feat] += 1
                
    output = [(key, val) for key, val in feat_appearances.items() if val>10] """
    feature_stats()
    print()


    