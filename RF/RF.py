
import os
import cv2
import time
import sklearn
from skimage import io
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import pickle


import matplotlib.pyplot as plt


class RandomForest:

    def __init__(self, data_pathway):
        
        self.path_0 = data_pathway

    def driver(self):

        dir_len = [] #Very quick to do this. Goes through 800k files in a couple seconds
        for character in os.listdir(self.path_0):
            character_path = os.path.join(self.path_0, character)
            n = 0 
            try:
                for file in os.listdir(character_path):      
                    #this will be used to determine size of the directories in order to balance the classes
                    n += 1
                dir_len.append(n)

            except NotADirectoryError:
                continue
        
        min_len = min(dir_len)

        train_images = []
        train_labels = {}
        train_numeric = []

        label_number = 0
        for character in os.listdir(self.path_0):
            if character == character: #Incase want to test only one specific directory/directories
                print(character)
                character_path = os.path.join(self.path_0, character)
                counter = 0 
                try:
                    for file in os.listdir(character_path): 
                        if counter < min_len:     
                            file = os.path.join(character_path, file)
                            images = cv2.imread(file)
                            images = cv2.resize(images, (100, 100))
                            images = cv2.normalize(images, None, 0, 1, cv2.NORM_MINMAX)
                            images = images.ravel() #flattens out the image into 100*100*3 
                            train_images.append(images)
                            counter += 1

                            if 'lower' in character:
                                train_labels[character[-1]] = label_number
                                train_numeric.append(label_number)
                            else:
                                train_labels[character] = label_number #this will be used to know the fruit with its numerical value
                                train_numeric.append(label_number)
                        else:
                            break

                except NotADirectoryError:
                    continue
                
                label_number += 1
                      

        # this code will be the main driver in order to train the model
        #Unbalanced was 800k, took a very long time (Testing:high accuracy could be biased?)

        
        #This line concatenates the train and test from how th original dataset was setup and splits it differently
        image_train, image_test, label_train, label_test = train_test_split(train_images, train_numeric, test_size=0.3, random_state=42)
                    

        tree_size_list = [10,100,1000]
        feature_size_list = [10,75,150,500]

        for i in tree_size_list:
            for j in feature_size_list:
                    model = RandomForestClassifier(n_estimators = i,
                                                    criterion = 'gini',
                                                    max_features= j,
                                                    max_depth= 100,
                                                    min_samples_split= 2,
                                                    min_samples_leaf= 1,
                                                    min_weight_fraction_leaf=0,
                                                    max_leaf_nodes= None,
                                                    min_impurity_decrease=0,
                                                    bootstrap = True,
                                                    oob_score = False,
                                                    n_jobs = -1,
                                                    random_state= None,
                                                    verbose = 100,
                                                    warm_start= False,
                                                    )

                    #This fits and predicts the train/test data
                    model.fit(image_train, label_train)

                    #save the model into directory 
                    with open('RFF' + str(j) + 'T' + str(i) + '_model.pickle', 'wb') as handle:
                        pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)

                #save the train_labels
                    with open('RFF' + str(j) + 'T' + str(i) + '_labels.pickle', 'wb') as handle:
                        pickle.dump(train_labels, handle, protocol=pickle.HIGHEST_PROTOCOL)

                    #save the test_set in case need to do analysis based off this model later on
                    with open('RFF' + str(j) + 'T' + str(i) + '_test_set.pickle', 'wb') as handle:
                        pickle.dump(image_test, handle, protocol=pickle.HIGHEST_PROTOCOL)

                    #save the test_nueric_labels to check against the test_set
                    with open('RFF' + str(j) + 'T' + str(i) + '_test_labels.pickle', 'wb') as handle:
                        pickle.dump(label_test, handle, protocol=pickle.HIGHEST_PROTOCOL)            



#Enter the image data pathway
path = '......../Data'
RF = RandomForest(path)
RF.driver()
 

