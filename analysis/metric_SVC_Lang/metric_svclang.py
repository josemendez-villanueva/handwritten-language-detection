# This file will contain the testing of the accuracy of the models that have been trained. Manually putting in the file names of trained models
# Jose Mendez-Villanueva
# 2021

import pickle
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import os
import cv2
import tensorflow as tf
from tensorflow.keras.models import model_from_json



##########################################################################################

#This will be run the models and save the plots to do the comparison
#Enter the pathway for the SVC Lang Folder, this will be used to get hte models

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
path = '..../handwritten-language-detection/SVC_Language/'
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# SVC
acc_list = []



#save the model into directory 
with open(path + 'SVC' + '_model.pickle', 'rb') as f:
    model = pickle.load(f)

#save the train_labels
with open(path + 'SVC'  + '_labels.pickle', 'rb') as f:
    labels = pickle.load(f)

#save the test_set in case need to do analysis based off this model later on
with open(path + 'SVC'  + '_test_set.pickle', 'rb') as f:
    test_set = pickle.load(f)

#save the test_nueric_labels to check against the test_set
with open(path + 'SVC'  + '_test_labels.pickle', 'rb') as f:
    test_label = pickle.load(f)     



prediction = model.predict(test_set)

accuracy =  accuracy_score(test_label ,prediction)

print(accuracy)

acc_list.append(accuracy)

cm = confusion_matrix(test_label, prediction)

report = classification_report(test_label ,prediction, output_dict = True)

import pandas as pd
df = pd.DataFrame(report).transpose()
df.to_csv('classification_report' + 'SVC' +  '.csv')       


plt.figure()
cmatrix = plt.matshow(cm) 
plt.title('Confusion matrix:' + 'SVC')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.colorbar()
plt.savefig('CM_SVC' + '.png')
plt.clf()
plt.close()



feat = ['SVC']
values = acc_list
plt.figure()
plt.bar(feat, values)
plt.ylabel('Accuracy')
plt.xlabel('SVC Method')
plt.title('Accuracy of SVC')
plt.savefig('Accuracy_of_SVC' + '.png')


##########################################################################################