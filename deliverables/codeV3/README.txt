

--------------------------------------------------------------------
comp 551 -- project 3
--------------------------------------------------------------------


Overall, the major packages we use in the project, consist of sklearn related packages, pytorch related packages. For image processing, we use skimage package. And some other common python packages such as numpy and pandas. Detailed as following:


data_load_and_preprocess_v17.py -- the script which does image data load and pre-process, support the functions such as image background denoise, separate touched digits, as well as crop the largest digit, for the usage of SVM. CNN models does not necessarily need image preprocessing. Mainly use skimage package.

cnn_vgg_customized_v5.py -- the customized Model for Vgg CNN which get the best prediction performance among all the different models we have, mainly using pytorch package.

cnn_resnet_normal_v0.py -- the Model for resnet CNN, using pytorch package as well as its resnet implementation.

svm_with_pca_v2.py -- the Model for SVM, mainly use sklearn package for SVM.


Instructions to replicate your results

1. run data_load_and_preprocess_v17.py to load, pre-process the modified mnist image data and generate the intermediate data in .csv format, for further fit/predict stage. It may take more than 1 hour to run and generate the intermediate pre-processed image data in .csv format.

2. run cnn_vgg_customized_v5.py or cnn_resnet_normal_v0.py, which could use the .pkl raw image data directly, or could also use intermediate pre-precessed image data. They both would generate the .csv format prediction of the test dataset for kaggle submission. It may spend very long time (several hours, at least) to run these deep learning models, and also need gpu cuda support, so please use these code in kaggle kernel rather than in local machine.  

3. run svm_with_pca_v2.py to use use the intermediate data and fit/predict with svc algorithm, which has options to support cross validation or generate .csv format, that can be submitted for the kaggle competition.


 

