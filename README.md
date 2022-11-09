### feature-selection-dual-ct
## feature-saving
- Run python3 data_process.py <br/>
- Inside Class DataProcess from *data_process.py*, function **_read_data_file()** read features from each patient file from data folder (./data/Features) and stored in a numpy array of dimension 21x4019 (**_num-energy-level x feature-dimension_**). For each energy level we get *95x21x4019* dimensional data (95 is the patient number) <br/>
- labels are also stored in **getLabel()** function. To read label for a particular end goal, masterfile is used. <br/>
- data is split in **train_val_split()** function <br/>
- set **dump=True** in *parameters.py* to save splitted set <br/>

## feature reduction and important feature selection
- Run python3 svm_classifier.py to apply PCA for each energy level data. Applying PCA on features from 65keV
(80x1x4019) gives 80x35 dimensional data.<br/>
- Features contributed most to each of the reduced dimension (**highest eigen value along each dimension**) is treated as important features. <br/>
- important feature will be written in a text file for each separate energy level (e.g. *./output/65.txt*) <br/>

## find common features
- Run feature_inspection.py to find out important features appearance over all energy levels <br/>
- feature_stats() will plot most frequent important feature vs energy level curve to observe the change. <br/>
- For now feature_stats() is only plotting change with one feature. It will be modified soon. <br/>