import os
import cv2

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import layers, models

import pickle
from sklearn.model_selection import train_test_split



import matplotlib.pyplot as plt


class Clustering:

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
        train_namelabel = []

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
                            # images = cv2.normalize(images, None, 0, 1, cv2.NORM_MINMAX)
                            images = cv2.cvtColor(images,cv2.COLOR_BGR2GRAY)
                            images = images.ravel()
                            #images = images.ravel() #flattens out the image into 100*100*3 
                            train_images.append(images)
                            counter += 1
                            

                            if 'lower' in character:
                                train_labels[character[-1]] = label_number
                                train_numeric.append(label_number)

                                for key, value in train_labels.items():
                                    if value == label_number:
                                        train_namelabel.append(key)
                                    else:
                                        continue
                            
                            else:
                                train_labels[character] = label_number #this will be used to know the fruit with its numerical value
                                train_numeric.append(label_number)

                                for key, value in train_labels.items():
                                    if value == label_number:
                                        train_namelabel.append(key)
                                    else:
                                        continue

                        else:
                            break

                except NotADirectoryError:
                    continue
                
                label_number += 1


        #TO SEE THE CLUSTERING OF THE NORMALIZED IMAGES
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        proj = pca.fit_transform(train_images)
        plt.scatter(proj[:, 0], proj[:, 1], c=train_numeric, cmap="Paired")
        plt.colorbar()
        plt.show()


#Enter your passway to to Data of Images
path = '......../Data'
Clust = Clustering(path)
Clust.driver()     