import cv2
import numpy as np
import math
from collections import defaultdict


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
        unlabeled_ids = []
        #image_name = "CT_"+str(i)+"kev"
        #print("image type: ", image_name)
        for patient_file in  os.listdir(self.data_folder):
            patient_id = patient_file[:patient_file.rindex('.')]
            #print("patient id: ",patient_id)
            df = pd.read_csv(os.path.join(self.data_folder, patient_file))
            #print(df.shape)
            #exit()
            patient_data = df.to_numpy()
            column_names = list(df.columns.values)
            column_names = column_names[37:]
            feature = patient_data[:, 37:]
            label = self.getLabel_Histologic_Grades(label_file, patient_id)
            if label == -1:
                unlabeled_ids.append(patient_id)
                continue
            labels.append(label)
            #print(feature.shape)
            #exit()
            feature_set.append(feature)

        feature_set = np.asarray(feature_set)
        labels = np.asarray(labels)
        print('feature set shape: ', feature_set.shape)
        print("labels: ", labels.shape)
        print("good: ", labels[labels==1].shape)
        print("bad: ", labels[labels==0].shape)
        print("unblabeled ids: ",unlabeled_ids, len(unlabeled_ids))
        exit()
        train_data, train_label, val_data, val_label =  self.five_fold_creation(feature_set, labels)
        #print(train_label, val_label)
        #print(train_label.shape, train_data.shape, val_label.shape, val_data.shape)
        folder = "histology"
        #exit()
        if params.dump:
            dump_file("val_folds", folder, val_data)
            dump_file("train_folds", folder, train_data)
            dump_file("val_label_folds", folder, val_label)
            dump_file("train_label_folds", folder, train_label)
            dump_file("features", folder, column_names)

    def five_fold_creation(self, feature_set, labels):
        class_data_per_fold = [[4, 4, 4, 3, 3], [10, 10, 10, 10, 10]]
        train_data_folds = defaultdict(list)
        val_data_folds = defaultdict(list)
        train_label_folds = defaultdict(list)
        val_label_folds = defaultdict(list)
        for i in range(2):  #Histological Grades -> 2
            data_class = feature_set[labels == i, :, :]
            label_class = labels[labels == i]
            indices = np.random.permutation(data_class.shape[0])
            data_class = data_class[indices, :, :]
            label_class = label_class[indices]
            start = 0
            selection = np.zeros(data_class.shape[0], dtype=bool)
            print("class: ", i)
            print("data: ", data_class.shape)
            print("label: ", label_class.shape)

            for j, data in enumerate(class_data_per_fold[i]):
                #print("Fold #",j+1)
                end = start + data
                selection[start:end] = True

                train_data_folds[j].extend(data_class[~selection, :, :])
                train_label_folds[j].extend(label_class[~selection])
                val_data_folds[j].extend(data_class[selection, :, :])
                val_label_folds[j].extend(label_class[selection])
                
                # print("train data: ", data_class[~selection, :, :].shape)
                # print("train label: ",label_class[~selection].shape)
                # print("val data: ",data_class[selection, :, :].shape)
                # print("val label: ",label_class[selection].shape)
                # print("selection: ",selection)
                start = end
                selection = np.zeros(data_class.shape[0], dtype=bool)

        return train_data_folds, train_label_folds, val_data_folds, val_label_folds 


    def train_val_split(self, feature_set, labels):
        train_set = []
        train_label = []
        val_set = []
        val_label = []

        split_index_class = [4, 4, 4, 3]

        
        #class_data_per_fold = [[4, 4, 4, 3, 3], [10, 10, 10, 10, 10]]
        for i in range(4):  #OC, OP, L/HP, Other
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
        #print('patient id:', patient_id)
        label_info = label_file.loc[label_file['Case ID'] == patient_id]
        #print("label info: ",label_info)
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

    def getLabel_Histologic_Grades(self, label_file, patient_id):
        #print('patient id:', patient_id)
        label_info = label_file.loc[label_file['Case ID'] == patient_id]
        #print("label info: ",label_info)
        label = -1

        in_situ = float(label_info.iloc[0]['In situ'])
        well_diff = float(label_info.iloc[0]['Well differentiated-low grade'])
        moderate_diff = float(label_info.iloc[0]['moderately differentiated'])
        poor_diff = float(label_info.iloc[0]['Poorly differentiated'])

        if not math.isnan(in_situ):
            label = 1
        elif not math.isnan(well_diff):
            label = 1
        elif not math.isnan(moderate_diff):
            label = 1
        elif not math.isnan(poor_diff):
            label = 0
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




