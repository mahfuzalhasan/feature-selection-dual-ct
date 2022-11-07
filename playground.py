import numpy as np

data = np.ones((95,21,4019))
labels = np.ones((95,))
labels_ext = np.expand_dims(labels, axis=1)
labels_ext = np.expand_dims(labels_ext, axis=2)
print(labels_ext.shape)
data_file = np.concatenate((data, labels_ext),axis=0)
print(data_file.shape)