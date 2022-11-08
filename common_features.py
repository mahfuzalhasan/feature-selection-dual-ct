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
    print("intersection ab")
    print(intersect_ab, len(intersect_ab))

    f = open(os.path.join(cfg.output_dir, '120.txt'), "r")
    data = f.read()
    c = list(set(data.split("\n")[:-1]))
    print("120: ",len(c))
    f.close()

    union_abc = set(a).union(b).union(c)
    print(union_abc, len(union_abc))
