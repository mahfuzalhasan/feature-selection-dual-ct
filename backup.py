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
        for i in range(55, 141, 5):
            feature_set = []
            labels = []
            column_names = []
            image_name = "CT_"+str(i)+"kev"
            print("image type: ", image_name)
            for patient_file in  os.listdir(self.data_folder):
                patient_id = patient_file[:patient_file.rindex('.')]
                df = pd.read_csv(os.path.join(self.data_folder, patient_file))
                #print(df.shape)
                #exit()
                patient_data = df.to_numpy()
                column_names = list(df.columns.values)
                column_names = column_names[37:]
                feature = patient_data[:, 37:]
                #print(feature.shape)
                feature_set.append(feature)
                #exit()
                label = self.getLabel(label_file, patient_id)
                labels.append(label)

            feature_set = np.asarray(feature_set)
            feature_set = feature_set.reshape(feature_set.shape[0], -1)
            
            print('feature set shape: ', feature_set.shape)
            labels = np.asarray(labels)

            train_data, train_label, val_data, val_label =  self.train_validation_split(feature_set, labels)
            print(train_label.shape, train_data.shape, val_label.shape, val_data.shape)
            folder = str(i)
            if params.dump:
                dump_file("val_data", folder, val_data)
                dump_file("train_data", folder, train_data)
                dump_file("val_label", folder, val_label)
                dump_file("train_label", folder, train_label)
                dump_file("features", folder, column_names)


    def train_validation_split(self, feature_set, labels):
        labels_ext = np.expand_dims(labels, axis=1)
        data_file = np.hstack([feature_set, labels_ext])
        OC_class = data_file[labels == 0,:]
        OP_class = data_file[labels == 1,:]
        L_class = data_file[labels == 2,:]
        Other_class = data_file[labels == 3,:]

        print("OC patients: ", OC_class.shape)
        print("OP Patients: ", OP_class.shape)

        print("L Patients: ", L_class.shape)
        print("Other Patients: ", Other_class.shape)
        exit()

        #np.random.shuffle(pos_class)
        #np.random.shuffle(neg_class)

        """ pos_class = pos_class[indices_1, :]
        neg_class = neg_class[indices_2, :]

        validation_set = np.concatenate((pos_class[:4], neg_class[:12]), axis=0)
        validation_label = validation_set[:, -1]

        train_set = np.concatenate((pos_class[4:], neg_class[12:]), axis=0)
        train_label = train_set[:, -1]

        validation_set = validation_set[:, :-1]
        train_set = train_set[:, :-1]

        return train_set, train_label, validation_set, validation_label """


    def getLabel(self, label_file, patient_id):
        label_info = label_file.loc[label_file['Case ID'] == patient_id]
        #print(patient_id)
        #print(label_info)
        label = -1
        #pos = float(label_info.iloc[0]['P16 pos'])
        #neg = float(label_info.iloc[0]['P16 neg'])

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




