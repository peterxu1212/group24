# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 02:24:43 2019

@author: PeterXu
"""


import warnings
warnings.filterwarnings('ignore')

import numpy as np
#from sklearn.datasets import fetch_mldata
import matplotlib.pyplot as plt


#from sklearn.decomposition import RandomizedPCA
from sklearn.decomposition import PCA as RandomizedPCA

from sklearn import datasets, svm, metrics
from sklearn import decomposition


from sklearn.model_selection import train_test_split


from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_predict

from sklearn.model_selection import cross_val_score


import time # computation time benchmark


import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


import logging
import logging.config


import os



#b_do_cross_validation = False
b_do_cross_validation = True


#b_do_for_submission = True
b_do_for_submission = False


print(os.listdir("../input"))

str_log_file_path = "../input/config/"

logging.config.fileConfig(str_log_file_path + 'logging.conf')





#logging.config.fileConfig('logging.conf')

# create logger
logger = logging.getLogger('Project3Group24')


logger.info("test log \n")


#input_data_folder = "../../../comp551_Modified_MNIST/output/"
#input_data_folder = "../../../project_3/digital_recognizer/input/"

#input_data_folder = "../input/digit-recognizer/"
#input_data_folder = "../input/pdmpdsv3/pdsv3/pdsv3/"

input_data_folder = "../input/pmdpdsv14/pdsv14/pdsv14/"


output_data_folder = input_data_folder


logger.info("input_data_folder = %s", input_data_folder)
logger.info(os.listdir(input_data_folder))


str_file_version_postfix = "_v14"
#str_file_version_postfix = ""

str_file_version_postfix = "_small" + str_file_version_postfix

str_train_file = 'train' + str_file_version_postfix + '.csv'
str_test_file = 'test' + str_file_version_postfix + '.csv'


logger.info(str_train_file)
logger.info(str_test_file)

train = pd.read_csv(input_data_folder + str_train_file)

test = pd.read_csv(input_data_folder + str_test_file)


logger.info(train.head())


i_partial_count = 1000

#b_partial = True
b_partial = False

X_set = None
Y_set = None

X_real_test_set = None

if b_partial:

    X_set = train.iloc[:, 1:]
    X_set = X_set[0:i_partial_count]

    #Y_set = train['label']
    Y_set = train.iloc[:, 0]    
    Y_set = Y_set[0:i_partial_count]
    
    
    X_real_test_set = test.iloc[:, 0:]
    X_real_test_set = X_real_test_set[0:i_partial_count]

else:

    X_set = train.iloc[:, 1:]
    Y_set = train.iloc[:, 0]
    #Y_set = train['label']
    
    X_real_test_set = test.iloc[:, 0:]



print("\n X_set.shape", X_set.shape)


print("\n Y_set.shape", Y_set.shape)


#logger.info(Y_set.shape)




x_train_set, x_val_set, y_train_set, y_val_set = train_test_split(X_set, Y_set, test_size = 0.20, random_state = 1)


if b_do_cross_validation:
    
    x_train_set = X_set    
    y_train_set = Y_set

else:
    if b_do_for_submission:
        
        x_train_set = X_set    
        y_train_set = Y_set
        
        

print("\n x_train_set.shape", x_train_set.shape)

print("\n y_train_set.shape", y_train_set.shape)





X_train, Y_train = np.float32(x_train_set) / 255., np.float32(y_train_set)




i_n_components = 100

pca = RandomizedPCA(n_components=i_n_components, random_state=32)

pca.fit(X_train)

# Plot explained variance ratio for the new dimensions.
logger.info("---------Variance explained-------------------")
logger.info(np.sum(pca.explained_variance_ratio_))


#quit()



# Fitting the new dimensions to the train-set.
X_train_ext = pca.fit_transform(X_train)

# New dimensions
logger.info("---------Train-set dimensions after PCA--------")
logger.info(X_train_ext.shape)


# Check how much time it takes to fit the SVM
start = int(round(time.time() * 1000))


# Fitting training data to SVM classifier.
# Fine-tuning parameters session included
# rbf, poly, linear and different values of gammma and C
classifier = svm.SVC(gamma=0.01, C=3, kernel='rbf')




if b_do_cross_validation:
    
    
    logger.info("---------(5) Cross validation accuracy--------")
    
    
    logger.info("start cross_val_score for svm")
    
    
    cv_score = cross_val_score(classifier, X_train_ext, Y_train, cv=5, verbose=10, n_jobs=3)
    
    
    logger.info("end cross_val_score for svm")
    
    logger.info(cv_score)
    
    
    
    mean_f1_from_cv = np.mean(cv_score)
    
    
    logger.info("\n\n mean_f1_from_cv = %f ", mean_f1_from_cv)
 
    

else:
        
    logger.info("start fit for svm")
    
    classifier.fit(X_train_ext, Y_train)
    
    logger.info("end fit for svm")


    # Using the last 10k samples as test set against the already trained cross-validated train-set.
    X_test, Y_test = np.float32(x_val_set) / 255., np.float32(y_val_set)
    
    # Fitting the new dimensions.
    X_test_ext = pca.transform(X_test)
    
    logger.info("---------Test-set dimensions after PCA--------")
    
    logger.info(X_test_ext.shape)
    expected = Y_test
    predicted = classifier.predict(X_test_ext)
    
    logger.info("--------------------Results-------------------")
    
    
    logger.info("\n %s ", metrics.classification_report(expected, predicted, digits=5))
    
    
    #logger.info(metrics.confusion_matrix(expected, predicted))
    
    


# End of time benchmark
end = int(round(time.time() * 1000))
logger.info("--SVM fitting finished in %d ms", (end - start))



if not b_do_cross_validation and b_do_for_submission:
    
    # real test set
    X_real_test = np.float32(X_real_test_set) / 255.
    
    # Fitting the new dimensions.
    X_real_test_ext = pca.transform(X_real_test)
    
    logger.info("---------Test-set dimensions after PCA--------")
    
    logger.info(X_real_test_ext.shape)
    #expected = Y_test
    real_test_pred = classifier.predict(X_real_test_ext)
    
    #logger.info("--------------------Results-------------------")
    
    
    #logger.info("\n %s ", metrics.classification_report(expected, predicted, digits=5))
    
    print("\n", type(real_test_pred), real_test_pred.shape)

    
    cur_time = int(time.time())

    print("cur_time = ", cur_time)

    str_file_name_tw = "csv/group24_" + "svm" + "_" + str(cur_time) + "_submission" + ".csv"


    
    df = pd.DataFrame(columns = ["Id", "Category"])
    
    real_test_pred_list = real_test_pred.tolist()
    
    i_index = 0
    
    for real_test_pred_item in real_test_pred_list:    
            
        list_row = []
        
        list_row.append(int(i_index))
        list_row.append(int(real_test_pred_item))
        
        df.loc[i_index] = list_row
        
        i_index += 1


    
    
    df.to_csv(output_data_folder + str_file_name_tw, index = False, header = True)


