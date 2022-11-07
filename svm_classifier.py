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

    


def important_feature_selection(train_features, val_features, initial_feature_names):
    # 10 samples with 5 features
    #train_features = np.random.rand(10,5)

    pca = PCA(n_components=0.95, svd_solver="full", random_state=1)
    pca.fit(train_features)
    X_pc = pca.transform(train_features)
    X_val = pca.transform(val_features)
    print("after pca feature shape: ",X_pc.shape, X_val.shape)

    # number of components
    #print("pca_components: ",pca.components_.shape)
    n_pcs= pca.components_.shape[0]
    #print("number of components: ", n_pcs)


    # get the index of the most important feature on EACH component
    # LIST COMPREHENSION HERE
    #most_important = [np.abs(pca.components_[i]).argmax() for i in range(n_pcs)]

    #initial_feature_names = ['a','b','c','d','e']
    # get the names
    #most_important_names = [initial_feature_names[most_important[i]] for i in range(n_pcs)]

    # LIST COMPREHENSION HERE AGAIN
    #dic = {'PC{}'.format(i): most_important_names[i] for i in range(n_pcs)}
    #print(dic)
    return X_pc, X_val

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
    mapper = {img_kev_list[i]:i for i in range(21)}
    input_keV = [65, 85, 120]
    #for keV in img_kev_list:
    print("image in keV: ",input_keV)
    indices = np.asarray([mapper[i] for i in input_keV])
    #indices = np.asarray([mapper[keV]])
    #indices = np.asarray(indices)

    folder = "global"
    
    X_train = read_file("train_data", folder)
    Y_train = read_file("train_label", folder)
    X_val = read_file("val_data", folder)
    Y_val = read_file("val_label", folder)

    #print(X_train.shape, X_val.shape)

    X_train = X_train[:, indices, :]
    X_val = X_val[:, indices, :]
    

    #print(X_train.shape, X_val.shape)
    #exit()

    X_train, Y_train = random_shuffle(X_train, Y_train)
    X_val, Y_val = random_shuffle(X_val, Y_val)
    initial_feature_names = read_file("features", folder)

    X_train = X_train.reshape(X_train.shape[0], -1)
    X_val = X_val.reshape(X_val.shape[0], -1)
    print(X_train.shape, X_val.shape)
    # Normalization
    scaler = MinMaxScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_val = scaler.transform(X_val)
    ######

    # type conversion of labels
    Y_train = Y_train.astype('int')
    Y_val = Y_val.astype('int')
    #print("data: ", X_train.shape, X_val.shape)
    #print('labels: ', Y_train.shape, Y_val.shape)
    ######

    ####

    # dimensionality_reduction
    #X_train, X_val = dimensionality_reduction(X_train, X_val)
    X_train, X_val = important_feature_selection(X_train, X_val, initial_feature_names)
    #print("dim reduced data: ", X_train.shape, X_val.shape)
    #dump_file("selected_feature", folder, dic)
    #exit()
    ##########################

    # MLPClassifier
    #print("##### MLP classifier ######")
    #clf = MLPClassifier(hidden_layer_sizes=(40, 20, 10), batch_size=8, max_iter=10000, learning_rate_init=0.0001, random_state=1)
    clf = MLPClassifier(hidden_layer_sizes=(100, 60, 30, 10), batch_size=8, max_iter=100000000, learning_rate_init=0.0001, random_state=1)
    clf.fit(X_train, Y_train)
    Y_predict = clf.predict(X_val)
    print('Y_val: ', Y_val)
    print('Y_pre: ', Y_predict)
    print("################################")
    ###############



""" # class weight
N = [(Y_train == 0).sum(), (Y_train == 1).sum()]
w = np.zeros((2,))
for i in range(2): #num_classes
    w[i] = (1/N[i])/((1/2)*((1/N[0])+(1/N[1])))
print("class weights: ", w)
##############



# Logistic regression check
print("##### Logistic Regression ######")
logisticRegr = LogisticRegression(solver = 'lbfgs', class_weight="balanced", max_iter=10000)
logisticRegr.fit(X_train, Y_train)
Y_predict = logisticRegr.predict(X_val)
print('Y_val: ', Y_val)
print('Y_pre: ', Y_predict)
###########################

C_list = [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 1e6, 1e7]

print("##### SVM ######")
for C in C_list:
    print("C: ", C)
    model = SVC(kernel="rbf", C=C, class_weight="balanced", max_iter=10000, random_state = 2022)
    # define evaluation procedure
    cv = RepeatedStratifiedKFold(n_splits=3, n_repeats=3, random_state=1)
    # evaluate model
    scores = cross_val_score(model, X_train, Y_train, scoring='roc_auc', cv=cv, n_jobs=-1)
    print('Mean ROC AUC: %.3f' % np.mean(scores))

    model.fit(X_train, Y_train)
    Y_predict = model.predict(X_val)

    print('Y_val: ', Y_val)
    print('Y_pre: ', Y_predict)

    #classifier.fit(X_train, Y_train)

    Y_predict = model.predict(X_val)

    print('Y_val: ', Y_val)
    print('T_pre: ', Y_predict) """
