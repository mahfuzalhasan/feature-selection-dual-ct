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


if __name__=="__main__":
    f = open(os.path.join(cfg.output_dir, '65.txt'), "r")
    data = f.read()
    a = list(set(data.split("\n")[:-1]))
    print("65: ",len(a))
    f.close()

    f = open(os.path.join(cfg.output_dir, '70.txt'), "r")
    data = f.read()
    b = list(set(data.split("\n")[:-1]))
    print("70: ",len(b))
    f.close()

    union_ab = set(a).union(b)
    print("union ab")
    print(union_ab, len(union_ab))
    print("intersection ab")
    intersect_ab = set(a).intersection(b)
    print(intersect_ab, len(intersect_ab))

    f = open(os.path.join(cfg.output_dir, '6570.txt'), "r")
    data = f.read()
    c = list(set(data.split("\n")[:-1]))
    print("65_70: ",len(c))
    f.close()

    print("comparison between 65 and 65_70")
    union_ac = set(c).union(a)
    print(union_ac, len(union_ac))
    intersect_ac = set(c).intersection(a)
    print("intersection ac")
    print(intersect_ac, len(intersect_ac))

    print("comparison between 70 and 65_70")
    union_bc = set(c).union(b)
    print(union_bc, len(union_bc))
    intersect_bc = set(c).intersection(b)
    print("intersection bc")
    print(intersect_bc, len(intersect_bc))

    print("comparison between (65+70) and 65_70")
    
    intersect = set(union_ab).intersection(c)
    print("intersection ab and c")
    print(intersect, len(intersect))
