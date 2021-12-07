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
from tensorflow.keras.models import model_from_json


##########################################################################################

#This will be run the models and save the plots to do the comparison
#Enter the pathway for the CNN Folder, this will be used to get hte models
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
path = '..../handwritten-language-detection/CNN/'
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# CNN Metrics
acc_list = []
optimizer = ['adam', 'adamax']
epoch_size_list = [3, 6, 10, 15]

for i in optimizer:

    for j in epoch_size_list:

        # load json and create model
        json_file = open(path + 'CNN' + i + 'E' + str(j) + '_model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)


        # load weights into new model
        loaded_model.load_weights(path + 'CNN' + i + 'E' + str(j) + '_model.h5')

        if 'adamax' in (path + 'CNN' + i + 'E' + str(j) + '_model.json'):             
            # evaluate loaded model on test data
            loaded_model.compile(optimizer='adamax',
                        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                        metrics=['accuracy'])

        else:
            # evaluate loaded model on test data
            loaded_model.compile(optimizer='adam',
                        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                        metrics=['accuracy'])



        #read the train_labels
        with open(path + 'CNN' + i + 'E' + str(j) + '_labels.pickle', 'rb') as f:
            labels = pickle.load(f)

        #read the test_set in case need to do analysis based off this model later on
        with open(path + 'CNN' + i + 'E' + str(j) + '_test_set.pickle', 'rb') as f:
            test_set = pickle.load(f)

        #read the test_nueric_labels to check against the test_set
        with open(path + 'CNN' + i + 'E' + str(j) + '_test_labels.pickle', 'rb') as f:
            test_label = pickle.load(f)     

        
        loss, acc = loaded_model.evaluate(test_set, test_label)

        prediction = loaded_model.predict(test_set)

        classes = np.argmax(prediction, axis = 1)

        acc_list.append(acc)


        cm = tf.math.confusion_matrix(
        test_label, classes, num_classes=52, weights=None, dtype=tf.dtypes.int32,
        name=None)

        
        fig = plt.figure()
        plt.matshow(cm) 
        plt.title('Confusion matrix:' + 'CNN' + i + 'E' + str(j))
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.colorbar()
        #plt.show()
        plt.savefig('Confusion_matrix' + 'CNN' + i + 'E' + str(j) + '.png')



        if j == 15:

            # plot the accuracies against each other
            fig = plt.figure()
            feat = [3, 6, 10, 15]
            values = acc_list
            plt.plot(feat, values, color='red', marker='o')
            plt.ylabel('Accuracies')
            plt.xlabel('Total Epochs')
            plt.title('Epoch Accuracies')
            plt.grid()
            plt.savefig('Epoch Accuracies' + str(i) + '.png')
            acc_list.clear()





##########################################################################################