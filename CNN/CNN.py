import os
import cv2

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import layers, models

import pickle
from sklearn.model_selection import train_test_split



import matplotlib.pyplot as plt


class CNN:

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
                            #images = images.ravel() #flattens out the image into 100*100*3 
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


        image_train = np.array(image_train)
        image_test = np.array(image_test)
        label_train = np.array(label_train)
        label_test = np.array(label_test)

        optimizer_list = ['adamax']
        #'adam'
        epoch_list = [6]

        #Do 10 for adamax

        model = models.Sequential([layers.Conv2D(100, (3,3), activation = 'relu', input_shape = (100, 100, 3)),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dense(52)
            ])


        print(model.summary())

        for i in optimizer_list:

            for j in epoch_list:

                model.compile(optimizer = i,
                                    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                                    metrics=['accuracy'])

            
                history = model.fit(image_train, label_train, epochs = j)    

                # serialize model to JSON
                model_json = model.to_json()
                with open('CNN' + i + 'E' + str(j) + '_model.json', "w") as json_file:
                    json_file.write(model_json)

                # serialize weights to HDF5
                model.save_weights('CNN' + i + 'E' + str(j) + '_model.h5')


                with open('CNN' + i + 'E' + str(j) + '_labels.pickle', 'wb') as handle:
                    pickle.dump(train_labels, handle, protocol=pickle.HIGHEST_PROTOCOL)


                ###########################################################################
                #save the test_set in case need to do analysis based off this model later on
                ################### Need this to do test accuracies and etc.###############
                with open('CNN' + i + 'E' + str(j) + '_test_set.pickle', 'wb') as handle:
                    pickle.dump(image_test, handle, protocol=pickle.HIGHEST_PROTOCOL)

                #save the test_nueric_labels to check against the test_set
                with open('CNN' + i + 'E' + str(j) + '_test_labels.pickle', 'wb') as handle:
                    pickle.dump(label_test, handle, protocol=pickle.HIGHEST_PROTOCOL)    

    


#Enter your pathway here
path = '............/Data'
Convolution = CNN(path)
Convolution.driver()                

