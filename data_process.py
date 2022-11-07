import cv2
import numpy as np
import math


import os


import torch
import torch.nn as nn
import torchvision


import pandas as pd
import csv
import os 
import pickle


import Config.configuration as cfg
import Config.parameters as params
from utils import *


class DataProcess(object):
    def __init__(self, data_folder):
        self.data_folder = data_folder
        self._read_data_file()

    def shuffle_along_axis(self, a, axis):
        idx = np.random.rand(*a.shape).argsort(axis=axis)
        return np.take_along_axis(a, idx, axis=axis)

    def _read_data_file(self):
        label_file = pd.read_excel(cfg.label_file, sheet_name = '3 - Masterfile P16 excl', engine='openpyxl')
        feature_set = []
        labels = []
        column_names = []
        #image_name = "CT_"+str(i)+"kev"
        #print("image type: ", image_name)
        for patient_file in  os.listdir(self.data_folder):
            patient_id = patient_file[:patient_file.rindex('.')]
            df = pd.read_csv(os.path.join(self.data_folder, patient_file))
            #print(df.shape)
            #exit()
            patient_data = df.to_numpy()
            column_names = list(df.columns.values)
            column_names = column_names[37:]
            feature = patient_data[:, 37:]
            label = self.getLabel(label_file, patient_id)
            labels.append(label)
            feature_set.append(feature)

        feature_set = np.asarray(feature_set)
        labels = np.asarray(labels)
        print('feature set shape: ', feature_set.shape)
        print("labels: ", labels.shape)

        train_data, train_label, val_data, val_label =  self.train_val_split(feature_set, labels)
        print(train_label, val_label)
        print(train_label.shape, train_data.shape, val_label.shape, val_data.shape)
        folder = "global"
        #exit()
        if params.dump:
            dump_file("val_data", folder, val_data)
            dump_file("train_data", folder, train_data)
            dump_file("val_label", folder, val_label)
            dump_file("train_label", folder, train_label)
            dump_file("features", folder, column_names)


    def train_val_split(self, feature_set, labels):

        train_set = []
        train_label = []
        val_set = []
        val_label = []

        split_index_class = [4,4,4,3]
        for i in range(4):  #4 classes
            data_class = feature_set[labels == i, :, :]
            label_class = labels[labels == i]
            indices = np.random.permutation(data_class.shape[0])
            data_class = data_class[indices, :, :]
            label_class = label_class[indices]

            train_set.extend(data_class[split_index_class[i]: ,:, :])
            train_label.extend(label_class[split_index_class[i]: ])

            val_set.extend(data_class[0:split_index_class[i], :, : ])
            val_label.extend(label_class[0:split_index_class[i]])

        train_set = np.asarray(train_set)
        val_set = np.asarray(val_set)

        train_label = np.asarray(train_label)
        val_label = np.asarray(val_label)
        return train_set, train_label, val_set, val_label 
        


    def getLabel(self, label_file, patient_id):
        #print('patient id:',patient_id)
        label_info = label_file.loc[label_file['Case ID'] == patient_id]
        label = -1

        OC = float(label_info.iloc[0]['OC'])
        OP = float(label_info.iloc[0]['OP'])
        L = float(label_info.iloc[0]['L/HP'])
        Other = float(label_info.iloc[0]['Other'])

        if not math.isnan(OC):
            label = 0
        elif not math.isnan(OP):
            label = 1
        elif not math.isnan(L):
            label = 2
        elif not math.isnan(Other):
            label = 3
        return label
            

    """ def dump_file(self, data_list, file_type):
        save_file_path = os.path.join(cfg.data_path, file_type+'.pkl')
        with open(save_file_path, 'wb') as f:
            pickle.dump(data_list, f)
        
    def image_file_list(self, data_frame):   
        image_files = [x for x in data_frame['compImageFile']]
        return image_files """

if __name__ == '__main__':
    data_process = DataProcess(cfg.data_folder)




