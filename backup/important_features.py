import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from utils import *
import Config.configuration as cfg


def draw_important_features(f_draw):
    #f_draw = read_file("common_feature", "Features")
    #check = [65, 70, 120]
    #for i in check:
    i=65
    image_name = "CT_"+str(i)+"kev"
    print("image type: ",image_name)
    feature_set = []
    #f_draw = f_draw[:7]
    for i in range(len(f_draw)):
        feature_set.append([])

    for patient_file in  os.listdir(cfg.data_folder):
        patient_id = patient_file[:patient_file.rindex('.')]
        print(patient_id)
        df = pd.read_csv(os.path.join(cfg.data_folder, patient_file))
        patient_data = df.loc[df['Unnamed: 0'] == image_name]
        for i, feature in enumerate(f_draw):
            #print(feature)
            value = float(patient_data[feature].values[0])
            #print(value)
            feature_set[i].append(value)
    print([len(l) for l in feature_set])
    feature_set_norm = [[i / sum(j) for i in j] for j in feature_set]
    x = [i+1 for i in range(95)]
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.title("A test graph")
    for i in range(len(feature_set_norm)):
        pt = feature_set_norm[i]
        plt.plot(x, pt, label = 'id %s'%i)
    plt.legend()
    plt.rc('legend', fontsize = 5)
    plt.show()
    plt.savefig('my_plot.png')
    plt.close()
    




if __name__=="__main__":
    features = []
    common_all = []
    check = [65, 70, 120]
    for i in check:
        folder = str(i)
        s_feat = read_file("selected_feature", folder)
        print(i,len(s_feat))
        features.extend(list(s_feat.values()))
        common_all.append(list(s_feat.values()))

    print("number of all features: ",len(features))
    common_features = list(set(features))
    dump_file("common_feature", "Features", common_features)
    print("common features: ",common_features,len(common_features))
    draw_important_features(common_features)




