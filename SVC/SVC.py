import os
import cv2

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import layers, models

import pickle
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier

import matplotlib.pyplot as plt
import time
start_time = time.time()


class SVC:

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


                            #PCA for 5 components
                            #First split the image into the three
                            B,G,R = cv2.split(images)

                            #PCA 
                            #reducing 100*100*1 into 200 dimensions
                            pca_B = PCA(n_components=5)
                            pca_B.fit(B)
                            images_B = pca_B.fit_transform(B)

                            
                            pca_G = PCA(n_components=5)
                            pca_G.fit(G)
                            images_G = pca_G.fit_transform(G)

                            pca_R = PCA(n_components=5)
                            pca_R.fit(R)
                            images_R = pca_R.fit_transform(R)

                            #Merge data back
                            #PCA image
                            images = cv2.merge((images_B, images_G, images_R))

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

        image_train, image_test, label_train, label_test = train_test_split(train_images, train_numeric, test_size=0.3, random_state=42)    


        from sklearn.svm import LinearSVC
        from sklearn.metrics import accuracy_score

        model = LinearSVC(multi_class='ovr', verbose = 100 )

        model.fit(image_train, label_train)
        

        #save the model into directory 
        with open('SVCpca20_model.pickle', 'wb') as handle:
            pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)

       #save the train_labels
        with open('SVCpca20_labels.pickle', 'wb') as handle:
            pickle.dump(train_labels, handle, protocol=pickle.HIGHEST_PROTOCOL)

        #save the test_set in case need to do analysis based off this model later on
        with open('SVCpca20_test_set.pickle', 'wb') as handle:
            pickle.dump(image_test, handle, protocol=pickle.HIGHEST_PROTOCOL)

        #save the test_nueric_labels to check against the test_set
        with open('SVCpca20_test_labels.pickle', 'wb') as handle:
            pickle.dump(label_test, handle, protocol=pickle.HIGHEST_PROTOCOL)         



        print("Program took", time.time() - start_time, "seconds to run")




#Enter the image data pathway
path = '...../Data'
svc = SVC(path)
svc.driver()     